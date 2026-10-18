# 全基因组结构扫描与聚类

本目录保存37℃ WT双重复的扫描评分、候选窗口、形态嵌入、AE/PCA六簇结果及已知结构召回指标。

`candidates.csv`包含294条初始候选；`unannotated_reproducible.csv`按扫描阶段的重复与注释条件筛出37条，和后续34条通过统一质量检查的未注释候选采用不同口径。`clusters.csv`及`novel_candidates.csv`采用簇中零已知成员、可复现成员至少5个的簇级判定。后者为空，仅说明该簇级规则下没有合格簇。

任务二候选新结构的统一操作性判据、六处发现和28条未注释已知样候选见[结构发现汇总](../discovery_candidate_analysis/findings/README.md)；294条完整证据与混合簇组成见[候选结构分析](../discovery_candidate_analysis/README.md)。簇级诊断和候选结构位置认定回答不同问题，不应将空的簇级结果理解为候选结构数量为零。

扫描实现：`scripts/discover_structures.py`；全量证据与结构发现汇总：`scripts/analyze_discovery_candidates.py`、`scripts/summarize_candidate_findings.py`。
