# 大肠杆菌 Micro-C 接触矩阵分析

本课程项目使用大肠杆菌 *E. coli* K-12 MG1655 的 Micro-C 接触矩阵，完成已知结构分类、未知结构候选发现、全基因组多条件可视化，以及**选做任务四方向二：已知结构在不同实验条件下的变化分析**。当前正式成果采用任务一至四。实验报告、答辩 PPT 和讲解视频均围绕这四项任务组织。

## 数据与实验范围

- 数据来源：[GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159)；登记 29 份 Cooler 矩阵、19 种条件标签，参考基因组为 `NC_000913.3`，原生分辨率 10 bp。
- 已知结构：344 条，其中 CHIN 250 条、OPCID 68 条、CHID 26 条。坐标及来源见 [data/annotations/README.md](data/annotations/README.md)、[structures.csv](data/annotations/structures.csv)。
- 条件差异分析：37℃ WT、44℃热应激、HNS_mut、FIS_mut，每种条件 2 个生物学重复，共 8 份矩阵。
- 大型原始 Cooler 文件主要保留在本地，GitHub 存储已提交的代码、注释、运行协议、评价表及代表性图像。文件来源与哈希见 [samples.csv](data/annotations/samples.csv)。

## 正式实验结果

| 任务 | 方法及范围 | 核心结果 | 结果入口 |
| --- | --- | --- | --- |
| 一：已知结构分类 | 344 条注释、56 个空间重叠组、5 折折外预测；三尺度重复一致性集成 CNN | Accuracy 0.7471，Macro-F1 0.6161；CHID 召回 10/26 | [分类结果](results/classification_fine/fine_consistent_ensemble_report.json) |
| 二：未知结构候选发现 | 全量候选分层、逐成员注释与形态分析、跨重复证据及混合簇分析 | 294条候选完整输出，118条基础质量通过；6个混合簇；未确认新形态类别。另行固定64预算检测评价的折外中心召回约0.2965 | [全量候选分析](results/discovery_candidate_analysis/README.md) |
| 三：多条件可视化 | WT/44℃/rif25 与 WT/HNS_mut/FIS_mut 两组条件，接触信号、平均信号、基因和已知结构四轨 | 两套图组各 465 张，连续覆盖全基因组 | [温度及药物轨道](results/multitrack/) · [突变轨道](results/multitrack_mutants/) |
| **四：已知结构的条件差异（选做）** | 对 344 条结构分别比较 44℃、HNS_mut、FIS_mut 与 WT；两重复、六种侧翼/背景敏感性 | **26 项稳健“结构 × 条件”比较，涉及 25 条不同注释、21 个扩展上下文组** | [条件差异证据汇总](results/structure_conditions/evidence_synthesis/README.md) |

### 选做任务四：主要发现

统一筛查包含 3 × 344 = 1,032 项结构—条件比较：

| 条件 | 稳健比较 | 目标区变化主导 | 主要结论 |
| --- | ---: | ---: | --- |
| 44℃ vs WT | 16 | 13 | 8 项局部富集增强、8 项减弱；目标区变化与背景变化可区分 |
| HNS_mut vs WT | 8 | 1 | 局部富集差异中有多项主要由侧翼变化解释 |
| FIS_mut vs WT | 2 | 2 | 两项的目标区接触密度与局部富集同向变化 |

在 44℃条件下，**OPCID_60** 的目标区接触密度增加约 +0.834 log2，**OPCID_56** 降低约 −1.176 log2；**OPCID_65** 虽然局部富集增加约 +1.299 log2，但目标区变化为 −0.154 log2，主要由侧翼降低约 −1.452 log2 解释。HNS_mut 下 **CHIN_229** 的局部富集约为 WT 的 2.14 倍，目标区仅增加约 +0.128 log2，不能解释为目标结构内部接触量翻倍。

完整的 26 行结果、重复方向、目标区/侧翼分解、已知注释重叠及证据分类见 [combined_robust_candidates.csv](results/structure_conditions/evidence_synthesis/combined_robust_candidates.csv)。热图与完整敏感性结果见 [44℃分析](results/structure_conditions/44C/README.md)和 [突变条件分析](results/structure_conditions/robustness/README.md)。

### 证据解释

每个条件只有两个生物学重复；经验重复噪声阈值用于筛选稳健的矩阵变化，不是正式 p 值或 FDR 校正。26 项比较共享 WT 样本，部分位于相同扩展上下文，因此不能视为 26 次独立发现。

GEO 样本数据量与重复编号**强烈支持**当前 WT 与 44℃系列 Cooler 对应同为 DNBSEQ-T7 的 t2 样本，但缺少 Cooler 文件与 GSM 的直接对应记录。HNS_mut 与 FIS_mut 则分别采用 DNBSEQ-G400 和 DNBSEQ-G50，而 WT 参考高度可能为 DNBSEQ-T7，突变组的条件效应仍可能混杂平台差异。结论限定为有重复和背景敏感性支持的**条件相关接触变化**，不推断直接调控机制。[平台核查与证据边界](results/structure_conditions/evidence_synthesis/README.md)

## 方法、图表和复现入口

- [实验方案及正式交付范围](docs/实验计划.md)
- [项目进度及交付检查](docs/进度.md)
- [任务四综合结果](results/structure_conditions/evidence_synthesis/)
- [任务一分类解释图](results/classification_fine/explanations/)
- [任务二全量候选与证据](results/discovery_candidate_analysis/README.md)
- [任务二逐候选对照图](results/discovery_candidate_analysis/heatmaps/)
- [任务三轨道图](results/multitrack/tracks/)

任务四的主要实现位于 `scripts/structure_conditions.py`、`scripts/structure_condition_shape.py`、`scripts/robustness_structure_conditions.py` 与 `scripts/summarize_structure_condition_evidence.py`。现有证据汇总可通过下列入口重新生成，原始矩阵相关分析须先确认本地 `data/raw/` 文件齐全：

```bash
.venv/bin/python scripts/summarize_structure_condition_evidence.py
.venv/bin/python -m pytest -q
```

**正式实验报告、PPT、讲解视频和提交包尚未制作**。交付前需要重新运行当前环境的测试、核实表格与图片引用，并补充成员及贡献信息。
