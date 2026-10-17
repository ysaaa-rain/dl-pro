"""Write the concise lab conclusion from computed tables, with no hand-filled metrics."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

import pandas as pd

from dlpro.method_review import load_config
from summarize_method_review import md_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method_review_final.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output"])
    review = root / "review_final"
    output = Path("docs/第五轮方法范式审查结论.md")
    if output.exists():
        raise FileExistsError(f"Preserve existing conclusion: {output}")
    decisions = json.loads((review / "decisions.json").read_text(encoding="utf-8"))
    novelty_status = json.loads((root / "novelty_corrected/status.json").read_text(encoding="utf-8"))
    prior_audit = json.loads((root / "train_prior_audit.json").read_text(encoding="utf-8"))
    integrity = json.loads((root / "final_audit.json").read_text(encoding="utf-8"))
    selected = decisions["sr"]["validation_selected"]
    sr_root = root / cfg["sr"]["result_subdir"]
    effects = pd.read_csv(sr_root / "evaluation/paired_intervals.csv")
    cols = ["psnr", "ssim", "oe_50bp_ssim", "oe_100bp_ssim"]
    selected_effect = effects[(effects.method == selected) & (effects.reference == "legacy_sparse_mse")
                              & (effects.split == "new_test") & effects.metric.isin(cols)]
    network = pd.read_csv(review / "sr_versus_validation_selected.csv")
    network = network[(network.method == cfg['sr']['primary_arm']) & (network.split == 'new_test')
                      & (network.comparison == 'cross_repeat') & network.metric.isin(cols)]
    expansion = effects[(effects.method == "context_expanded_shape") & (effects.reference == "context_sparse_shape")
                         & (effects.split == "new_test") & (effects.comparison == "cross_repeat") & effects.metric.isin(["psnr", "oe_100bp_ssim"])]
    novelty = pd.read_csv(root / 'novelty_corrected/novelty_summary.csv')
    joint = novelty[(novelty.method == 'joint_multiscale') & (novelty['repeat'] == 1) & (novelty.negative == 'allowed_known')]
    cluster = pd.read_csv(root / 'novelty_corrected/cluster_quality.csv')
    tests_text = (root / 'legacy_audit_final/pytest.txt').read_text(encoding='utf-8').strip()
    tests = int(re.search(r'(\d+) passed', tests_text).group(1))
    date = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    cross100 = selected_effect[(selected_effect.comparison == 'cross_repeat') & (selected_effect.metric == 'oe_100bp_ssim')].iloc[0]
    crosspsnr = selected_effect[(selected_effect.comparison == 'cross_repeat') & (selected_effect.metric == 'psnr')].iloc[0]
    neural_shape = network[network.metric == 'oe_100bp_ssim'].iloc[0]
    if neural_shape.low > 0:
        neural_text = '网络对该指标有额外收益，其他指标仍须联合判断。'
    elif neural_shape.high < 0:
        neural_text = '该网络的100bp形态指标低于验证选中的统计分支，不能把总体形态提升归因于网络。'
    else:
        neural_text = '未建立网络额外100bp形态收益的稳健证据。'
    prefix = '../' + root.as_posix()
    text = f'''# 任务二与任务五：第五轮结论

日期：{date}。完整结果和逐例表由程序生成于[自动实验报告]({prefix}/review_final/REPORT.md)。协议、纠错时间线见[根本瓶颈与固定协议](第五轮根本瓶颈与固定协议.md)。全部结果属于已观察的同一基因组内的回顾性空间检验，外部生物学验证尚未完成。

## 任务二

完成条件背景残差、浅层自监督新颖性和跨尺度/重复联合三类路线，保留形态、PCA等对照。原始{novelty_status['known_total']}条注释完整保留历史基线复现；最终新分区评价{novelty_status['known_eligible']}条，各重叠上下文和两个重复遵守同一空间分区，共{novelty_status['fits']}个留类/折拟合。边界排除和尾区纠错均留有清单及旧产物。

核心瓶颈是已知检出与未知形态判别的目标错位、嵌套标注导致参考形态不独立，以及可用独立参考不足。隐藏CHIN并剔除共享上下文后，CHID不能继续提供独立参考。新评分降低部分背景误报，却没有建立跨类别可靠的新颖性识别。联合方法的绝对AUROC如下：

{md_table(joint[['hidden_type','n_hidden','n_negative','auc','low','high']])}

CHIN相对旧评分的差值有局部改善，但绝对区间包含随机水平；CHID没有同步改善。完整配对差值、误报、AUPRC、候选预算和不同类型召回均保留，未挑选有利类别汇报。

固定六簇与HDBSCAN的合格暂定形态簇累计为{int(cluster.provisional_shape_cluster.sum())}。候选可以具有真实重复支持，仍需证明其与已知形态不同；未注释、独立窗口数量增加不能直接构成新类别。继续保留原support-aware morphology操作方案。

## 任务五

全部固定训练臂、两种子和简单基线均已运行。概率预测与输出质量解耦；对称矩阵只在原生上三角计入唯一接触项；零源块仍可接受概率监督，最终输出严格保持源质量。输出为期望图像计数，不冒充测得的整数读段。160bp作为40bp整数倍，只作守恒对照；学习收益考察50/100bp。

统一验证在测试前选择了`{selected}`。相对同一新划分、同一归一化下重训的正式模型结构，新保留空间组跨重复100bp形态SSIM差为{cross100.delta:+.6f}，组区间[{cross100.low:+.6f},{cross100.high:+.6f}]；PSNR差为{crosspsnr.delta:+.6f} dB，区间[{crosspsnr.low:+.6f},{crosspsnr.high:+.6f}]。

完整的新保留组同/跨重复取舍如下。此处的对照均在相同新划分、同一评估器内比较；旧汇报的绝对PSNR口径另列历史复现。

{md_table(selected_effect[['comparison','metric','delta','low','high']])}

同重复50bp形态存在退化。保留统计先验为共享跨重复重建的可选分支，原正式默认方案不切换。独立空间块证据来自同一个历史暴露基因组，不称外部泛化。

### 数据扩展的实际贡献

相同模型、损失和计算步数下，扩大合法训练采样带来非常小的改善：

{md_table(expansion[['metric','delta','low','high']])}

它只支持本轮固定计算预算下的结论，不证明所有训练规模或充分收敛预算的上限。

### 神经网络的额外贡献

预定混合目标网络相对验证选中分支的差值为：

{md_table(network[['metric','delta','low','high']])}

{neural_text} 均匀分配也保留在对照中，因此不能把简单重分配能取得的分数改善写成新的局部结构恢复。当前证据支持改善条件分配的偏差/噪声取舍，尚未证明恢复不可复现的10bp局部细节。

## 已完成、验证与边界

已实现并运行：新分区与原件提取、三类新颖性路线、固定预算/开放集评价、两类聚类、五臂两种子训练、统一验证选择和完整测试、可直接应用的守恒推理分支。

已验证：{tests}项测试通过；{len(integrity['checks'])}项本轮整体完整性检查通过；训练分位数/距离先验/OE背景均从训练数据重算匹配，距离先验独立累积的最大差为{prior_audit['independently_accumulated_profile_max_absolute_error']:.3e}；实际推理示例与冻结评估逐像素一致。来源/数据/源码归档、模型哈希和真实退出状态完整保留。

两处科学实现纠错已完成：规则网格归一化偏差、未分配尾区误入训练。原产物保留并标记淘汰；缓存复核中的变量遮蔽错误及修复也留有退出日志。旧审计写死候选/簇结果的断言已改为按正式规则核验。

待完成的科学证据：外部更深覆盖的匹配观测、候选的盲法独立生物验证，以及能区分嵌套类型的新颖性标签定义。低块内重复相关是噪声诊断，不是不可突破的理论性能上限。正式Word、PPT、视频和提交包继续暂停。本轮修改保留本地。

## 使用与复现

配置在`configs/`，按基础协议、归一化纠错和最终空间纠错继承，避免重复定义方法参数。新运行使用新的output/cache目录；已有运行不会被覆盖。可审计的逐次命令、实际退出状态和源码在`{root.as_posix()}/runs/`。

直接应用验证选择的分支：先设置`PYTHONPATH=src`，运行`python scripts/reconstruct_review_sr.py --input-low <40bp均值矩阵.npy> --output <新预测.npz>`。示例输入、输出和零差复核在`review_final/demo_*`、`validated_prior_demo.npz`及`prediction_smoke_check.json`。

完整修改清单见[文件清单]({prefix}/review_final/change_manifest.csv)，失败/淘汰记录见[运行状态]({prefix}/review_final/run_status.csv)，全部主指标及区间见自动实验报告和CSV。所有生物学新发现、外部验证及全尺度无退化的说法均不在本轮证据内。
'''
    output.write_text(text, encoding='utf-8', newline='\n')
    readme = Path('README.md')
    current = readme.read_text(encoding='utf-8')
    banner = f'> 当前实验状态（{date}）：任务二、五根本方法审查完成，见[第五轮结论](docs/第五轮方法范式审查结论.md)。任务二仍无可靠新形态簇；任务五训练距离先验有有限跨重复收益及同重复形态取舍。正式报告继续暂停，下方历史结果保留作为对照。'
    readme.write_text(banner + '\n' + current.split('\n', 1)[1], encoding='utf-8', newline='\n')
    progress = Path('docs/进度.md')
    previous = progress.read_text(encoding='utf-8')
    heading = f'## {date}：第五轮根本方法审查'
    if heading in previous:
        raise ValueError('Do not duplicate the completed progress section')
    progress.write_text(previous.rstrip() + f'\n\n{heading}\n\n任务二完成三类路线、{novelty_status["fits"]}个留类空间拟合及{novelty_status["known_eligible"]}条完整评价；聚类未获得合格新形态簇。任务五完成固定五臂两种子和统一基线，验证选择{selected}，新保留组跨重复100bp形态差{cross100.delta:+.6f}、PSNR差{crosspsnr.delta:+.6f}dB；同重复50bp形态退化。{tests}项测试及{len(integrity["checks"])}项整体完整性检查通过，先验和背景重算核验、推理示例均完成。纠错/失败产物完整保留，新分支保留为有取舍的实验推荐，正式默认不切换。详细结果见[第五轮结论](第五轮方法范式审查结论.md)及`{root.as_posix()}/review_final/REPORT.md`。正式报告、PPT和提交包继续暂停，本轮未推送远端。\n', encoding='utf-8', newline='\n')
    print('Conclusion and repository indexes generated from actual result tables', flush=True)


if __name__ == '__main__':
    main()
