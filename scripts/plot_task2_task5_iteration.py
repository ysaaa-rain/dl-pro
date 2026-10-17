"""Standalone scientific figures for the bounded task-2/task-5 iteration."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlpro.coverage_support import coarse_support


OUT = Path("results/iteration_review")


def candidate_figure():
    table = pd.read_csv("results/discovery_directional_cv/candidate_context_audit.csv")
    table = table[table.independent_context_member].reset_index(drop=True)
    assert len(table) == 4
    scan = pd.read_csv("data/processed/scan_windows.csv")
    ids = [int(scan.index[scan.center == c][0]) for c in table.center]
    with np.load("results/discovery_directional_cv/fold0_scores.npz") as source:
        expected = source["expected"][0]
    distance = abs(np.arange(128)[:, None] - np.arange(128)[None, :])
    images, supports = {}, {}
    for repeat in (1, 2):
        with np.load(f"data/processed/scan_rep{repeat}.npz") as source:
            images[repeat] = source["raw"][ids, 0]
        with np.load(f"data/processed/coverage_rep{repeat}.npz") as source:
            supports[repeat] = coarse_support(source["coverage"], table.center.to_numpy())[:, 0] >= .8
    fig, axes = plt.subplots(4, 2, figsize=(8.5, 13), layout="constrained")
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("#d4d4d4")
    for i, row in enumerate(table.itertuples()):
        common = supports[1][i] & supports[2][i]
        valid = common[:, None] & common[None, :] & (distance >= 2)
        for repeat in (1, 2):
            oe = images[repeat][i] / expected[distance]
            amplitude = oe[valid].mean()
            shape = np.ma.masked_where(~valid, np.log1p(oe / max(amplitude, 1e-8)))
            ax = axes[i, repeat-1]
            im = ax.imshow(shape, origin="lower", cmap=cmap, vmin=0, vmax=3,
                           extent=(-6.4, 6.4, -6.4, 6.4))
            ax.set(title=f"{row.candidate_id} · WT重复{repeat} · {row.center:,} bp",
                   xlabel="相对中心 (kb)", ylabel="相对中心 (kb)")
        axes[i, 0].text(.02, .98, f"支持像素相关 r={row.wt_oe_correlation_supported:.3f}",
                        transform=axes[i, 0].transAxes, va="top", color="white", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=.5, label="幅度归一化 log(1+O/E)；灰色为排除像素")
    fig.suptitle("4 个未注释复现候选：尚无合格新结构簇", fontsize=15)
    fig.savefig(OUT / "task2_candidates.png", dpi=150)
    plt.close(fig)


def comparison_figure():
    task2 = pd.read_csv("results/discovery_directional_cv/paired_summary.csv")
    sr = pd.read_csv("results/sr_conditional/test_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    styles = [("old_morphology", "原形态", "#65758b"),
              ("combined_validation_selected", "方向形态（验证选择）", "#d98b32"),
              ("forest_known_control", "方向森林（已知检测）", "#287e78")]
    ax = axes[0, 0]
    for key, label, color in styles:
        for repeat in (1, 2):
            part = task2[(task2.type == "all") & (task2.repeat == repeat)].sort_values("budget")
            ax.plot(part.budget, part[key], marker="o", color=color,
                    linestyle="-" if repeat == 1 else "--", label=f"{label} · 重复{repeat}")
    ax.set(title="A 任务二：相同候选预算的中心召回", xlabel="候选预算", ylabel="344条注释的召回率",
           xticks=[64, 128], ylim=(0, .75))
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[0, 1]
    labels = ["CHID", "CHIN", "OPCID"]
    x = np.arange(3)
    part = task2[(task2.budget == 64) & (task2.repeat == 2)].set_index("type").loc[labels]
    weighted = pd.read_csv("results/discovery_type_weight_control/per_type.csv")
    weighted = weighted[(weighted.budget == 64) & (weighted.repeat == 2)
                        & (weighted.recipe == "type_weighted")].set_index("type").loc[labels]
    for offset, values, label, color in [(-.23, part.old_morphology, "原形态", "#65758b"),
                                        (0, part.forest_known_control, "方向森林", "#287e78"),
                                        (.23, weighted.recalled, "类型权重对照", "#d98b32")]:
        ax.bar(x+offset, values, width=.22, label=label, color=color)
    ax.set(title="B 任务二：类型之间存在取舍", xticks=x, xticklabels=labels,
           ylabel="重复2召回率 · 64候选", ylim=(0, .8))
    ax.legend(fontsize=9)
    methods = ["projected_bicubic", "dense_multiscale", "cross_repeat_target", "conditional_rep1", "conditional_rep2"]
    names = ["约束插值", "原多尺度", "投影重复目标", "计数损失\n重复1目标", "计数损失\n重复2目标"]
    colors = ["#8e99a6", "#4169a1", "#d98b32", "#9c6daf", "#287e78"]
    native = sr[sr.split == "cross_repeat_native"].set_index("method").loc[methods]
    ax = axes[1, 0]
    gain = native.psnr - native.loc["projected_bicubic", "psnr"]
    ax.bar(np.arange(len(methods)), gain, color=colors)
    for i, value in enumerate(gain):
        ax.text(i, value + .002, f"{value:+.3f}", ha="center", fontsize=9)
    ax.set(title="C 任务五：跨重复PSNR相对约束插值", ylabel="增益 (dB)",
           xticks=np.arange(len(methods)), xticklabels=names)
    ax.axhline(0, color="#444444", linewidth=.7)
    ax = axes[1, 1]
    for offset, split, label, marker in [(-.08, "same_repeat", "同重复", "o"),
                                         (.08, "cross_repeat_native", "跨重复原生", "s")]:
        part = sr[sr.split == split].set_index("method").loc[methods]
        difference = part.oe_100bp_masked_ssim - part.loc["dense_multiscale", "oe_100bp_masked_ssim"]
        ax.plot(np.arange(len(methods)) + offset, difference, marker=marker, label=label,
                linestyle="--", linewidth=.7)
    ax.set(title="D 任务五：100 bp形态相对原多尺度模型", ylabel="掩码O/E SSIM差值",
           xticks=np.arange(len(methods)), xticklabels=names)
    ax.axhline(0, color="#444444", linewidth=.7)
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("任务二与任务五：固定对照及负面权衡\n同研究、已观察空间位置上的回顾性评价", fontsize=15)
    fig.savefig(OUT / "iteration_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10})
    candidate_figure()
    if Path("results/sr_conditional/test_summary.csv").exists():
        comparison_figure()
