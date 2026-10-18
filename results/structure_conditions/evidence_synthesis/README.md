# 条件差异证据汇总

在既定的稳健性筛选规则下，比较 WT 与 44℃、HNS_mut、FIS_mut 条件的已知结构接触变化，并统一分解结构目标区与清洁侧翼背景的接触密度变化。

## 结果

共保留26项“结构注释 × 条件”比较，涉及25条不同注释、21个扩展上下文组。比较行不能视为独立发现：所有条件复用WT参考矩阵，部分注释处于同一扩展上下文，且OPCID_10在HNS_mut与FIS_mut中各出现一次。

| 条件 | 比较数 | 扩展上下文组 | 目标区四组跨重复方向一致 | 目标区同向且变化至少0.25 log2 | 目标区主导 | 侧翼主导或背景驱动 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 44C | 16 | 14 | 16 | 13 | 13 | 3 |
| HNS_mut | 8 | 7 | 5 | 2 | 1 | 3 |
| FIS_mut | 2 | 2 | 2 | 2 | 2 | 0 |

其中，HNS_mut另有1项表现为目标区与侧翼均变化、但侧翼变化幅度更大；另3项的目标区方向未能在全部四个跨条件重复组合中保持一致。OPCID_10在两种突变条件中的目标区接触密度下降均大于侧翼下降。44C的16项中，13项由目标区变化主导；CHIN_69、CHIN_79的局部富集变化主要由侧翼变化解释，OPCID_65则是目标密度基本稳定、侧翼密度明显下降导致相对富集升高。

代表性案例在两份重复矩阵的热图中也能看到相同的大体形态：44C下OPCID_60的目标区信号增强、OPCID_56减弱；OPCID_65的局部富集升高来自侧翼信号下降，不能解释成目标结构增强。HNS_mut下OPCID_10、FIS_mut下OPCID_15的目标区信号均减弱；不过突变条件与WT的测序平台不同，相关图像只能作为探索性支持。

每项变化满足数值恒等式：

`局部富集 log2 变化 = 目标区密度 log2 变化 − 侧翼密度 log2 变化`

重算后的最大绝对分解误差为 `2.7 × 10⁻¹⁵`。目标区“主导”使用的规则是：方向在四个跨条件重复组合中一致，与局部富集变化同向，目标区绝对变化至少0.25 log2，且目标区变化幅度大于侧翼变化幅度。

## 44C样本来源与平台

GEO记录中，WT和44C各有t1、t2两组样本；t1为NextSeq 2000，t2为DNBSEQ-T7。GEO系列级Cooler按rep1/rep2提供，不含GSM编号。系列Cooler接触总数为3.53亿至3.76亿；对应t1 SRA样本仅有3,871万至7,148万spots，矩阵接触数高出5.3至9.1倍，可排除t1样本作为这些矩阵的一对一来源。t2的SRA样本约有5.95亿至6.37亿spots，Cooler接触总数占其0.55至0.61，且rep编号吻合。因此系列级WT与44C矩阵高度可能对应t2的两组生物学重复，均采用DNBSEQ-T7。NCBI未提供文件级GSM映射，故这一对应关系是强证据推断，不是直接确认。

WT、44C的t2样本同平台比较，使44C的温度差异解释比突变比较更可信。FIS_mut样本为DNBSEQ-G50，HNS_mut为DNBSEQ-G400；对应WT参考高度可能为DNBSEQ-T7，且没有同平台WT对照。局部富集对整矩阵乘法缩放不敏感，但不能完全消除平台造成的距离或形态偏差。因此，HNS_mut和FIS_mut的结构变化应视为跨平台的探索性信号，不能单独归因于突变。

## 空间重叠

21个扩展上下文组中有5组含有多项候选比较。OPCID_10在HNS_mut与FIS_mut的两行是同一已知区间在两个条件下的比较；另有四组包含邻近但区间不直接重叠的不同注释。不同候选注释之间没有基因组坐标直接重叠。另一方面，8项候选注释的目标坐标与其他已知结构注释相交，所以背景排除和扩展上下文分组仍然重要。

只有两个生物学重复，经验重复噪声筛选不是置换检验、FDR或家族错误率控制。已知结构坐标曾用于前序分析；结果属于回顾性稳健性比较，不是独立验证。

## 文件

- `combined_robust_candidates.csv`：26项比较的局部富集、目标区与侧翼密度、重复方向、重叠注释和解释标签。
- `candidate_context_groups.csv`：候选比较所在的扩展上下文组及其空间依赖。
- `candidate_pair_relationships.csv`：同一注释跨条件重复或处于同一上下文组的比较对。
- `44c_geo_sample_platform_audit.csv`：8个GEO样本的t1/t2分组、平台、SRA spots和对应Cooler总数。
- `condition_platform_comparability.csv`：WT、44C、HNS_mut、FIS_mut的测序平台对照。
- `series_cooler_integrity.csv`：WT与44C系列Cooler的本地坐标、分辨率和矩阵总数检查。
- `summary.json`：机器可读的合并计数和恒等式误差。

重建上述表格：

```bash
.venv/bin/python scripts/summarize_structure_condition_evidence.py
```

元数据来源：[GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159)、[t1 44C样本 GSM8394771](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394771)、[t2 WT样本 GSM8394774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394774)、[t2 44C样本 GSM8394777](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394777)。SRA spots来自各样本关联的SRA RunInfo记录，链接见`44c_geo_sample_platform_audit.csv`。
