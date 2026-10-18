"""Generate comparisons and decisions directly from actual immutable outputs."""

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlpro.method_review import DEFAULT_CONFIG, load_config, write_json
from evaluate_allocation_review import interval


LABELS = dict(morphology_baseline="Morphology", conditional_anomaly="Conditional anomaly",
              conditional_novelty="Conditional novelty", pca_novelty="PCA novelty",
              paired_novelty="Paired representation", joint_multiscale="Joint (uses both repeats)")


def md_table(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.5f}"
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def paired_auc_difference(labels, left, right, groups, cfg):
    labels, groups = np.asarray(labels, bool), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    def differences(score):
        return (score[pos, None] > score[neg]).astype(float) + .5 * (score[pos, None] == score[neg])
    diff = differences(np.asarray(left)) - differences(np.asarray(right))
    wins = np.zeros((len(unique), len(unique)))
    np.add.at(wins, (inverse[pos, None], inverse[neg][None]), diff)
    pc = np.bincount(inverse[pos], minlength=len(unique))
    nc = np.bincount(inverse[neg], minlength=len(unique))
    weight = np.random.default_rng(cfg["seed"]).multinomial(len(unique), np.full(len(unique), 1/len(unique)), cfg["bootstrap_draws"])
    numerator = np.einsum("bi,ij,bj->b", weight, wins, weight)
    denominator = (weight @ pc) * (weight @ nc)
    estimates = numerator[denominator > 0] / denominator[denominator > 0]
    lo, hi = np.quantile(estimates, [.025, .975])
    return dict(delta=float(diff.mean()), low=float(lo), high=float(hi), groups=len(unique), valid_draws=len(estimates))


def summarize_novelty(cfg, out):
    root = Path(cfg["output"]) / cfg["discovery"].get("result_subdir", "novelty")
    points = pd.read_csv(root / "point_scores.csv")
    summary = pd.read_csv(root / "novelty_summary.csv")
    estimates = []
    for (hidden, repeat), part in points.groupby(["hidden_type", "repeat"]):
        baseline = part[part.method == "morphology_baseline"]
        for method in sorted(set(part.method) - {"morphology_baseline"}):
            paired = part[part.method == method].merge(baseline, on=["item_id", "actual_type", "group"], suffixes=("_new", "_base"), validate="one_to_one")
            for negative in ("allowed_known", "background"):
                unknown = paired.actual_type == hidden
                mask = unknown | (paired.actual_type.eq("background") if negative == "background" else (~unknown & ~paired.actual_type.eq("background")))
                subset = paired[mask]
                result = paired_auc_difference(subset.actual_type.eq(hidden), subset.score_new, subset.score_base, subset.group, cfg)
                estimates.append(dict(hidden_type=hidden, repeat=repeat, method=method, negative=negative, **result))
    difference = pd.DataFrame(estimates)
    difference.to_csv(out / "novelty_paired_auc_differences.csv", index=False)
    recall = pd.read_csv(root / "budget_recall.csv")
    recall = recall[recall.hidden_type == recall.actual_type]
    records = []
    for keys, part in recall.groupby(["hidden_type", "repeat", "method", "global_equivalent_budget"]):
        result = interval(part.recalled.astype(float).to_numpy(), part.group.to_numpy(), cfg)
        records.append(dict(zip(["hidden_type", "repeat", "method", "global_equivalent_budget"], keys),
                            n=len(part), recall=result.pop("delta"), **result))
    pd.DataFrame(records).to_csv(out / "novelty_budget_recall_summary.csv", index=False)
    methods = list(LABELS)
    hidden_types = sorted(summary.hidden_type.unique())
    fig, axes = plt.subplots(len(hidden_types), 2, figsize=(13, 10), constrained_layout=True)
    for row, hidden in enumerate(hidden_types):
        for column, negative in enumerate(("allowed_known", "background")):
            ax = axes[row, column]
            for repeat, offset, color in ((1, -.12, "#3068a0"), (2, .12, "#d07c25")):
                panel = summary[(summary.hidden_type == hidden) & (summary.negative == negative) & (summary["repeat"] == repeat)].set_index("method").loc[methods]
                ax.errorbar(panel.auc, np.arange(len(methods)) + offset,
                            xerr=[panel.auc-panel.low, panel.high-panel.auc], fmt="o", markersize=4,
                            capsize=2, color=color, label=f"WT repeat {repeat}")
            ax.set_yticks(np.arange(len(methods)), [LABELS[m] for m in methods])
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.axvline(.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("AUROC (100 kb group bootstrap 95% interval)")
            n_hidden = int(panel.n_hidden.iloc[0])
            n_negative = int(panel.n_negative.iloc[0])
            ax.set_title(f"Hidden {hidden} (n={n_hidden}) vs {negative} (n={n_negative})")
            ax.grid(axis="x", alpha=.2)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.savefig(out / "novelty_comparison.png", dpi=160)
    plt.close(fig)
    primary = difference[(difference.method == "joint_multiscale") & (difference.negative == "allowed_known")]
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    gate = bool((primary.low > 0).all() and status["full_success_no_missing_folds"])
    return summary, difference, gate


def summarize_sr(cfg, out):
    sr = cfg["sr"]
    root = Path(cfg["output"]) / sr.get("result_subdir", "sr")
    evaluation = root / "evaluation"
    frame = pd.read_csv(evaluation / "per_window.csv")
    intervals = pd.read_csv(evaluation / "paired_intervals.csv")
    choice = json.loads((evaluation / "validation_selection.json").read_text(encoding="utf-8"))
    candidate = choice["selected_method"]
    main = intervals[(intervals.split == "new_test") & (intervals.comparison == "cross_repeat")
                     & (intervals.method == sr["primary_arm"]) & (intervals.reference == sr["primary_comparator"])
                     & (intervals.metric == sr["primary_metric"])]
    primary_pass = bool(len(main) and main.low.iloc[0] > 0)
    selected_pass = candidate != "legacy_sparse_mse"
    selected_evidence = []
    for split in ("new_test", "historical_test"):
        for comparison in ("same_repeat", "cross_repeat"):
            for metric, floor in (("psnr", -sr["max_psnr_drop_db"]), ("ssim", -sr["max_ssim_drop"])):
                values = intervals[(intervals.split == split) & (intervals.comparison == comparison)
                                   & (intervals.method == candidate) & (intervals.reference == "legacy_sparse_mse") & (intervals.metric == metric)]
                selected_pass &= bool(len(values) and values.delta.iloc[0] >= floor)
            if comparison == "cross_repeat":
                values = intervals[(intervals.split == split) & (intervals.comparison == comparison)
                                   & (intervals.method == candidate) & (intervals.reference == "legacy_sparse_mse") & (intervals.metric == sr["primary_metric"])]
                selected_pass &= bool(len(values) and values.low.iloc[0] > 0)
                selected_evidence.extend(values.to_dict("records"))
    checks = json.loads((evaluation / "protocol_and_checks.json").read_text(encoding="utf-8"))
    selected_pass &= checks["all_nonnegative"] and checks["constrained_max_block_relative_error"] < checks["relative_tolerance"]
    metrics = ["psnr", "ssim", "oe_50bp_ssim", "oe_100bp_ssim", "oe_160bp_ssim", "raw_nrmse", "stabilized_poisson_deviance_per_count"]
    average = frame.groupby(["split", "comparison", "method"])[metrics].mean().reset_index()
    average.to_csv(out / "sr_unified_summary.csv", index=False)
    against_selected = []
    for (split, comparison), part in frame.groupby(["split", "comparison"]):
        point_means = part.groupby(["method", "group", "start", "source_repeat"])[metrics].mean().reset_index()
        reference = point_means[point_means.method == candidate]
        for method in sorted(set(point_means.method) - {candidate}):
            pair = point_means[point_means.method == method].merge(reference, on=["group", "start", "source_repeat"], suffixes=("_method", "_selected"), validate="one_to_one")
            for metric in metrics:
                estimates = interval((pair[metric+"_method"]-pair[metric+"_selected"]).to_numpy(), pair.group.to_numpy(), cfg)
                against_selected.append(dict(split=split, comparison=comparison, method=method,
                                              reference=candidate, metric=metric, **estimates))
    pd.DataFrame(against_selected).to_csv(out / "sr_versus_validation_selected.csv", index=False)
    core = ["uniform", "projected_bicubic", "train_distance_prior", *sr["arms"]]
    labels = {name: name.replace("_", " ") for name in core}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, comparison, metric in ((axes[0, 0], "cross_repeat", "psnr"), (axes[0, 1], "cross_repeat", "oe_100bp_ssim"),
                                   (axes[1, 0], "same_repeat", "psnr"), (axes[1, 1], "same_repeat", "oe_50bp_ssim")):
        data = frame[(frame.split == "new_test") & (frame.comparison == comparison)]
        means = data.groupby(["method", "group", "start", "source_repeat"])[metric].mean().reset_index()
        reference = means[means.method == "legacy_sparse_mse"]
        effects = []
        for position, method in enumerate(core):
            pair = means[means.method == method].merge(reference, on=["group", "start", "source_repeat"], suffixes=("_method", "_reference"), validate="one_to_one")
            effect = interval((pair[metric+"_method"]-pair[metric+"_reference"]).to_numpy(), pair.group.to_numpy(), cfg)
            effects.append(effect)
            color = "#c2742d" if method == candidate else "#3879aa"
            ax.hlines(position, effect["low"], effect["high"], color=color, linewidth=2)
            ax.plot(effect["delta"], position, "o", color=color, markersize=5)
        ax.set_yticks(np.arange(len(core)), [labels[m] for m in core])
        ax.invert_yaxis()
        low, high = min(e["low"] for e in effects), max(e["high"] for e in effects)
        width = max(high-low, 1e-6)
        for position, effect in enumerate(effects):
            ax.text(effect["high"]+width*.025, position, f"{effect['delta']:+.4f}", va="center", fontsize=8)
        ax.set_xlim(low-width*.06, high+width*.28)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{comparison} / {metric}")
        ax.set_xlabel("Paired difference vs retrained legacy; 100 kb group 95% interval")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Reserved spatial groups / orange = validation-selected statistical prior", fontsize=12)
    fig.savefig(out / "sr_comparison.png", dpi=160)
    plt.close(fig)
    selected = frame[(frame.split == "new_test") & (frame.comparison == "cross_repeat") & (frame.method == sr["primary_arm"])]
    base = frame[(frame.split == "new_test") & (frame.comparison == "cross_repeat") & (frame.method == sr["primary_comparator"])]
    per_seed = []
    for seed in sr["train_seeds"]:
        left = selected[selected.seed == seed]
        right = base[base.seed == seed]
        pair = left.merge(right, on=["group", "start", "source_repeat"], suffixes=("_left", "_right"), validate="one_to_one")
        for metric in ("psnr", "ssim", sr["primary_metric"]):
            per_seed.append(dict(seed=seed, metric=metric, **interval((pair[metric+"_left"]-pair[metric+"_right"]).to_numpy(), pair.group.to_numpy(), cfg)))
    pd.DataFrame(per_seed).to_csv(out / "sr_primary_seed_stability.csv", index=False)
    primary_pass &= all(row["delta"] > 0 for row in per_seed if row["metric"] == sr["primary_metric"])
    # The predeclared safety limits apply to the primary comparison too.
    for comparison in ("same_repeat", "cross_repeat"):
        panel = intervals[(intervals.split == "new_test") & (intervals.comparison == comparison)
                          & (intervals.method == sr["primary_arm"]) & (intervals.reference == sr["primary_comparator"])]
        for metric, floor in (("psnr", -sr["max_psnr_drop_db"]), ("ssim", -sr["max_ssim_drop"])):
            primary_pass &= panel.loc[panel.metric == metric, "delta"].iloc[0] >= floor
    plot_probes(root, out, checks["scale"], sr)
    tradeoffs = intervals[(intervals.method == candidate) & (intervals.reference == "legacy_sparse_mse")
                          & (intervals.comparison == "same_repeat") & (intervals.metric == "oe_50bp_ssim")]
    return average, dict(validation_selected=candidate, selected_passes_spatial_cross_gate=bool(selected_pass),
                         primary_hypothesis_passes=bool(primary_pass), primary_interval=main.to_dict("records"),
                         selected_evidence=selected_evidence, selected_same_repeat50bp_effects=tradeoffs.to_dict("records"),
                         recommendation_scope="Optional shared-cross-repeat reconstruction branch; full same-repeat local fidelity not established",
                         count_checks=checks)


def plot_probes(root, out, scale, sr):
    probes = sorted((root / "evaluation").glob("probe_*.npz"))
    training_expected = np.load(root / "normalization.npz")["expected_100"]
    for path in probes:
        source = np.load(path)
        fig, axes = plt.subplots(2, 6, figsize=(18, 6), constrained_layout=True)
        images = {key: source[key] for key in ("source", "target", "bicubic", "prior", "primary")}
        images["input40"] = np.repeat(np.repeat(source["low"], sr["factor"], axis=0), sr["factor"], axis=1)
        names = ["input40", "source", "target", "bicubic", "prior", "primary"]
        for col, name in enumerate(names):
            a = images[name]
            native_im = axes[0, col].imshow(np.clip(np.log1p(a)/scale, 0, 1), cmap="magma", vmin=0, vmax=1, origin="lower")
            axes[0, col].set_title(name.replace("primary", sr["primary_arm"]).replace("_", " "), fontsize=9)
            n, f = len(a)//10, 10
            pooled = a[:n*f, :n*f].reshape(n, f, n, f).mean((1, 3))
            d = abs(np.arange(n)[:, None] - np.arange(n)[None])
            oe = np.log1p(pooled/training_expected[d])
            oe[d < 2] = 0
            oe_im = axes[1, col].imshow(oe, cmap="magma", vmin=0, vmax=2, origin="lower")
            axes[1, col].set_xlabel("100 bp bin")
        axes[0, 0].set_ylabel("10 bp normalized log count")
        axes[1, 0].set_ylabel("100 bp training-fitted log O/E")
        fig.colorbar(native_im, ax=axes[0, :], label="Clipped log1p count / training scale", shrink=.8)
        fig.colorbar(oe_im, ax=axes[1, :], label="log1p O/E (shared training expectation)", shrink=.8)
        fig.suptitle(path.stem + " / fixed random selection before inspecting test gains")
        fig.savefig(out / (path.stem+".png"), dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--novelty-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output"])
    out = args.output if args.output is not None else root / "review"
    out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        summarize_sr(cfg, out)
        print("Actual-output figures and paired effect tables refreshed", flush=True)
        return
    novelty, differences, novelty_pass = summarize_novelty(cfg, out)
    if args.novelty_only:
        print("novelty summary complete", flush=True)
        return
    sr, sr_decision = summarize_sr(cfg, out)
    novelty_root = root / cfg["discovery"].get("result_subdir", "novelty")
    diagnostic_root = root / cfg.get("diagnostics_subdir", "diagnostics")
    candidate = pd.read_csv(novelty_root / "candidate_quality_summary.csv")
    clusters = pd.read_csv(novelty_root / "cluster_quality.csv")
    counts = pd.read_csv(diagnostic_root / "sr_train_val_summary.csv")
    known = pd.read_csv(root / "data/known_eligibility.csv")
    known["evaluation_eligible"] = known.eligible & known.new_fold.ge(0)
    eligibility = known.groupby(["type", "evaluation_eligible"]).size().unstack(fill_value=0).reset_index()
    runs = []
    for path in sorted((root / "runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "command" in record:
            disposition = ("superseded: grid quantile bias; stopped before SR test" if path.stem == "train" else
                           "superseded: unassigned partial-tail training inputs" if path.stem in {"novelty", "candidate_audit", "cluster_audit", "novelty_review"} else
                           "cache verification failed: manifest variable shadowing; fixed in next attempt" if path.stem == "prepare_verify_compatible" else
                           "see stage checks and limitations")
            runs.append(dict(stage=path.stem, status=record["status"], scientific_disposition=disposition, exit_code=record["exit_code"], elapsed_seconds=record.get("elapsed_seconds"), command=" ".join(record["command"])))
    pd.DataFrame(runs).to_csv(out / "run_status.csv", index=False)
    primary = differences[(differences.method == "joint_multiscale") & (differences.negative == "allowed_known")]
    compact = novelty.pivot(index=["hidden_type", "method"], columns=["negative", "repeat"], values="auc")
    compact.columns = [f"{a}_rep{b}" for a,b in compact.columns]
    selected_candidate = sr_decision["validation_selected"]
    lines = ["# 任务二与任务五：方法范式审查和冻结对照结果", "",
             "本文件由真实逐例输出自动生成。基线版本f44d348；所有新位置仍属于历史已观察的同一E. coli基因组，未获得外部生物学验证。正式报告、PPT、视频和提交包未改。", "",
             "## 基线与正确性", "",
             "见baseline_reproduced/baseline_register.json和reproduction_checks.json。任务二原344条、56空间组、64/128预算形态结果重新推理；任务五六分支和正式投影SubpixelSR检查点重新推理。历史训练未重跑，不把检查点推理复现写成从头训练复现。", "",
             "修复：旧审计不再写死4个候选/0个簇，改为按数据验证QC、独立性与簇规则。新训练二维网格统计偏差在测试前发现并中止，失败原件和源码留在sr/及runs/train*；正确实验为sr_corrected/。精确训练像素直方图计算归一化，并有完整数组quantile与保留样本隔离检查。", "",
             "## 任务二", "", "### 数据与根本瓶颈", "",
             "原始344条全部保留历史复现。新空间留类检验仅使用完整25.6kb输入落在单一完整100kb块的样本。原novelty/运行曾把无折号尾区默认加入训练，评价覆盖登记也相差一条；现novelty_corrected/明确排除无折号输入，保留原运行，作为已观察基因组内的纠错复算。几何边界先固定，尾区缺陷在图审查时发现，不能把纠正运行当全新独立测试。逐项几何信息见data/known_eligibility.csv，最终evaluation_eligible还要求new_fold>=0。越界样本的结果不得由本检验外推。", "", md_table(eligibility), "",
             "隐藏CHIN时，剔除共享上下文后已知参考极少，而且没有可与CHIN上下文分离的CHID参考；嵌套标注不是互斥的新形态类别。fits.json记录各折背景、参考、阈值数据的实际索引；diagnostics_corrected/novelty_fitting_audit.json核验未使用隐藏上下文或验证/测试折拟合。", "",
             "### 方法与完整对照", "",
             "A为多变量条件背景残差及已知参考距离，B为PCA与跨重复浅层自监督表示，C为多尺度/两重复最弱证据联合。所有权重/维数/背景规则均在新评分前固定；无隐藏类型标签参与训练或阈值选择。联合方法已经使用两个重复，其重复相关不称独立验证。", "",
             md_table(compact.reset_index()), "", "主方案joint相对原形态的已知/留出分离AUROC差及空间组区间：", "",
             md_table(primary[["hidden_type", "repeat", "delta", "low", "high", "groups"]]), "",
             "全部AUPRC、样本量、背景误报、固定预算逐例召回、配对AUROC区间、跨重复及nuisance相关保存在novelty_corrected/和review_final/的CSV。64为全726格点等效预算，分配到空间合格测试格点后实际选择数另列，不与历史全基因组64候选混用。", "",
             "### 候选、聚类及取舍", "",
             md_table(candidate[(candidate['repeat']==1)&(candidate.global_equivalent_budget==cfg['discovery']['global_budgets'][0])][["hidden_type","method","selected","repeat_qc","independent"]]), "",
             "质检数量增加不能证明新颖性改善。错位重复对照按训练背景密度/稀疏/支持匹配；整体O/E相关和距离层相关分开。所有候选采用同一完整上下文排除和25.6kb独立间隔。未计算全基因组显著性p值；相关窗口和选择偏差不能用朴素多重检验忽略。", "",
             md_table(clusters.assign(non_noise_group=clusters.cluster.ge(0), noise_set=clusters.cluster.lt(0)).groupby("algorithm").agg(non_noise_groups=("non_noise_group","sum"), noise_sets=("noise_set","sum"), provisional_shape_clusters=("provisional_shape_cluster","sum")).reset_index()), "",
             "表中簇数是多个留类/重复/预算/方法组合累计的聚类实例，不能算作独立生物学发现。聚类是固定逐窗描述符上的探索性分组，含全部候选及已知上下文，未对测试候选拟合PCA或调节最小成员数。暂定形态簇不等于新生物类别。",
             f"固定新颖性替换门槛通过：{novelty_pass}。" + ("评分仍需独立生物验证。" if novelty_pass else "保留原support-aware morphology操作方案；新方法不能仅凭更少误报或更多未注释窗口替换。"), "",
             "## 任务五", "", "### 数据和数学目标", "",
             "低输入为16个10bp图像像素的均值，输出为非负期望图像计数，块内和等于均值×16。对称Cooler像素不是独立读段。显式模型先预测概率，再乘源质量；零源块概率仍有跨重复监督梯度，但最终输出仍为零。条件目标对原生上三角唯一接触项计数、单独归一化粗对角块，作为依赖块的复合目标。", "",
             md_table(counts[["split","source_repeat","native_zero_fraction","input_zero_block_fraction","target_mass_in_zero_source_fraction","normalized_within_block_correlation"]]), "",
             "新训练为21块231窗，稀疏对照42窗；新验证5块10窗、新保留6块12窗，原8测试块16位置继续保留。所有新模型从头训练，两个seed、每臂320步；不能与旧960步模型作纯架构归因。实际感受野5720bp小于完整10240bp输入，全部输入严格落在各自空间块。两个重复是配对观测，不是两个独立基因组。全库深度是固定实验元数据；训练分位数、距离先验与O/E期望仅在新训练位置拟合。", "",
             "低块内重复相关不是模型可恢复信息的严格理论上限，只说明原生计数监督噪声很大；条件均值仍可能从更多合法数据学到。零源块在严格守恒下不能给另一重复恢复正质量，这是单独的约束冲突。训练量扩展结论限定在相同320步计算预算，不能外推成任何收敛预算下扩充均无效。", "",
             "### 统一结果", "",
             "以下为两个seed指标平均，非预测集成；简单基线只算一次。所有方法使用同一输入、目标和评估。50/100bp为非对齐形态指标；160bp是40bp的整数倍，其保真由守恒保证，只作物理对照，不能当作学习收益。", ""]
    for split in ("new_test", "historical_test"):
        for comparison in ("same_repeat", "cross_repeat"):
            lines.extend([f"#### {split} / {comparison}", "", md_table(sr[(sr.split==split)&(sr.comparison==comparison)].drop(columns=["split","comparison"])), ""])
    lines += ["### 保留决策与未解决问题", "",
              f"测试前统一验证选择：{selected_candidate}。主假设mixed相对expanded-context通过预定形态及原生安全门槛：{sr_decision['primary_hypothesis_passes']}。验证候选通过新空间与历史块跨重复复核：{sr_decision['selected_passes_spatial_cross_gate']}。", "",
              "通过的是预定跨重复100bp、PSNR和原生SSIM安全屏障，不代表全部形态尺度同时获益。验证候选的同重复50bp效应如下；保留为共享跨重复重建的可选分支，原操作默认方案不切换。脚本scripts/reconstruct_review_sr.py可直接应用测试前已选定的分支，输出非负期望计数及守恒检查。", "",
              md_table(pd.DataFrame(sr_decision['selected_same_repeat50bp_effects']).reindex(columns=["split","delta","low","high"])), "",
              "网络相对验证选中基线的完整配对差值见review_final/sr_versus_validation_selected.csv；均匀分配和训练距离先验均保留，避免把可由简单重分配得到的形态收益归因于神经网络。", "",
              "所有差值和组级bootstrap见sr_corrected/evaluation/paired_intervals.csv；两seed主对照见review_final/sr_primary_seed_stability.csv。样本扩展比较context_sparse_shape与context_expanded_shape；概率/混合目标比较allocation_expanded_cross与allocation_expanded_mixed，不把架构和训练量同时变化当成单因素收益。", "",
              "仅在验证选择及冻结测试均满足预定要求时列为推荐候选；否则保留原正式投影首轮CNN。输出稀疏计数、跨重复总量差和同基因组历史暴露限制仍然存在，不能声称恢复了无重复支持的10bp细节或获得外部验证。", "",
              "## 运行状态、文件与复现", "", md_table(pd.DataFrame(runs)[["stage","status","scientific_disposition","exit_code","elapsed_seconds"]]), "",
              "已实现/已运行/已验证以各stage实际记录、CSV、检查点和测试日志为准。被中止的旧归一化训练是失败/淘汰运行，不混入正确实验。新阶段保存完整源码ZIP、源码/配置/数据清单哈希、设备、命令、起止时间和实际退出状态；早期阶段的源归档清单单列。", "",
              "运行入口：先配置PYTHONPATH=src，使用项目虚拟环境运行scripts/run_method_review.py。最终复现各阶段统一使用--config configs/method_review_final.json；summarize另传--report-output以指定报告目录。基础与中间配置只用于追溯保留的旧运行。每阶段写独立日志，已有记录不会被覆盖；重跑应使用新output目录，且准备数据时也指定对应cache目录。", "",
              "进一步投入优先级：外部/更深覆盖的匹配原始数据；独立候选的盲法人工形态复核与多条件重复；能区分嵌套类型的新标签任务定义。继续在同一历史测试集搜索更好数字不能替代这些证据。", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_json(out / "decisions.json", dict(novelty_gate=novelty_pass, sr=sr_decision,
                                           production_choices_modified=False, biological_validation_available=False))
    process = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--short"], capture_output=True, text=True, encoding="utf-8")
    (out / "changed_files.txt").write_text(process.stdout, encoding="utf-8", newline="\n")
    print("automatic report and comparison figures generated", flush=True)


if __name__ == "__main__":
    main()
