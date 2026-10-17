> 当前实验状态（2026-10-17）：任务二、五新一轮固定对照已完成，见[第四轮结果](docs/第四轮实验结果.md)与[实验状态](docs/进度.md)。任务二仍无合格新结构簇，任务五仍有形态与原生误差取舍。报告暂停编写，已有Word为暂存草稿；下方历史结果保留作为对照。

# Micro-C 接触矩阵深度学习实验

已完成必做任务一至三及选做任务五的第一轮计算。数据来自GSE272159：29个矩阵、19种条件标签；结构注释344条。完整原始数据13.79 GB留在本地，小体积数据、处理窗口、模型和结果纳入Git，`.gitignore` 不隐藏数据。

## 第一轮结果与边界

| 实验 | 实测结果 | 限制 |
|---|---|---|
| 分类 | 56个重叠组、五折344条折外预测，准确率73.8%，macro-F1 0.599 | CHID召回34.6%；同研究内部验证 |
| 发现 | 726扫描窗口、294去重候选、37个未注释跨重复窗口 | 0个合格新簇；同预算随机召回优于检测器 |
| 多轨道 | 37C、44C、rif25，各两个重复；465张图连续覆盖全基因组 | 局部±10 kb接触信号，无ICE平衡，不推断机制 |
| 超分辨 | 40→10 bp，三个种子均改善原生PSNR/SSIM；49个完整上下文下游准确率65.3%→69.4% | O/E SSIM双三次0.991、CNN0.962，结构对比度反而退化 |

报告为 [可编辑Word](output/实验报告_第一轮计算结果.docx)，原始指标位于 results 各实验目录；8项测试通过。macOS Accelerate矩阵运算警告保留，独立非BLAS计算复核预测、PCA和聚类有限且一致；跨平台仍应重验。没有新的生物学结构结论。成员信息、PPT视频与最终提交包尚待完成。

## 环境与复跑

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,experiment,report]'
.venv/bin/python -m pytest -q
# 首次恢复全部原始矩阵（约13.79 GB含压缩与解压）
python3 scripts/download_geo.py --include-wt
.venv/bin/python scripts/prepare_geo.py
.venv/bin/python scripts/resolve_reference.py
# 已提交缓存有SHA256核验；去掉参数可从原始矩阵重新提取
.venv/bin/python scripts/prepare_experiments.py --reuse-cache
.venv/bin/python scripts/train_classifier.py
.venv/bin/python scripts/crossvalidate_classifier.py
.venv/bin/python scripts/discover_structures.py
.venv/bin/python scripts/multitrack.py
.venv/bin/python scripts/sr_expanded.py
.venv/bin/python scripts/coverage_qc.py
.venv/bin/python scripts/validate_experiments.py
.venv/bin/python scripts/summarize_results.py
.venv/bin/python scripts/report_figures.py
.venv/bin/python scripts/build_experiment_report.py
```

MPS可用则用于SR，其他训练使用CPU；不同机器不保证逐位一致。报告叙述是已审阅的第一轮快照，重跑后除刷新表格外还需复核文字与排版。作者注释复用固定提交和哈希，不随main自动漂移。

已提交 known/scan 缓存和SR压缩计数分片。单个大文件 raw_counts.npy、reconstruction_seed42.npy 与大原始矩阵保留本地，运行脚本可恢复；不为规避上限拆分大原始文件。发现旧强度对照保留模型与数值，重复热图仅留本地；主方法294张候选热图和465张轨道图纳入Git。提交前运行 `python3 scripts/check_staged_sizes.py`。

基础CLI支持audit、inspect和window，局部预览为局部O/E，不能代替正式归一化；基础CLI不支持跨起点，但正式实验稀疏读取器支持环状起点及末端短bin面积校正。

详见 [进度](docs/进度.md)、[设计与验收](docs/实验计划.md)。来源、下载大小和本地SHA256见 results/geo_download_summary.json、results/geo_matrix_audit.json；发布者未给独立SHA256，不能声称与官方哈希吻合。

## 第二轮改进（2026-10-14）

[第二轮Word报告](output/实验报告_第二轮改进对照.docx)与[对照记录](docs/第二轮实验.md)保留第一轮负面结果。

- 任务二改用固定64/128候选预算，验证集选择形态评分。64候选原测试中心召回35.4%，随机均值17.8%、第一轮AE18.5%。但12个空间组的重采样差值区间包含零，尚不能声称稳健优于随机；10个未注释复现候选，仍0个合格新簇。
- 任务五增加40bp计数一致性投影、100bp O/E损失与消融。对旧CNN直接投影，100bp O/E SSIM约0.870→0.915；WTrep1 PSNR22.205→22.032，结构改善有图像误差代价。新增三种子训练没有超过旧CNN加投影。200bp保真按约束保证，不能算独立学习证据；下游准确率没有保留第一轮69.4%的提升。
- 新增WT/HNS_mut/FIS_mut的465张轨道图，独立缓存与目录，不覆盖第一轮图组。正式使用10矩阵，19矩阵仅审计，详见data/annotations/sample_usage.csv。
- 10项测试通过，真实计数守恒、缓存条件拒绝、PCA/聚类独立复核和轨道连续覆盖已检查。所有原测试位置此前已观察，本轮是回顾性比较。

```bash
.venv/bin/python scripts/discovery_refinement.py
.venv/bin/python scripts/sr_refinement.py
.venv/bin/python scripts/sr_projection_ablation.py
.venv/bin/python scripts/multitrack.py --conditions 37C HNS_mut FIS_mut --output results/multitrack_mutants
.venv/bin/python scripts/summarize_refinement.py
.venv/bin/python scripts/refinement_figures.py
.venv/bin/python scripts/validate_refinement.py
.venv/bin/python scripts/build_refinement_report.py
```

## 空间验证与模型对照

正式使用11个矩阵（NoLig作为单样本控制），18个仅审计。分类新增50bp输入、重复一致性、多种子、跨度辅助匹配对照、核方法和几何森林。任务二采用五折背景隔离、固定预算随机对照及训练召回校准；完整上下文与支持像素筛选剩4个独立探索候选，未确认新结构簇。大感受野SR相对计数投影旧CNN略改善原生指标，100bp结构收益区间包含零。

可按以下顺序恢复新增实验。各训练脚本会重写其结果目录；模型选择规则和缓存来源在对应protocol/sources文件中。报告脚本不属于当前执行流程。

```bash
.venv/bin/python scripts/prepare_fine_windows.py
.venv/bin/python scripts/prepare_fine_scan.py
.venv/bin/python scripts/classification_balanced_cv.py
.venv/bin/python scripts/classification_shape_cv.py
.venv/bin/python scripts/classification_kernel_cv.py
.venv/bin/python scripts/classification_hybrid_cv.py
.venv/bin/python scripts/classification_fine_cv.py
.venv/bin/python scripts/classification_span_cv.py
.venv/bin/python scripts/audit_classification_labels.py
.venv/bin/python scripts/explain_fine_classifier.py
.venv/bin/python scripts/discovery_spatial_cv.py
.venv/bin/python scripts/discovery_support_cv.py
.venv/bin/python scripts/discovery_recall_operating_point.py
.venv/bin/python scripts/candidate_condition_validation.py
.venv/bin/python scripts/candidate_negative_control.py
.venv/bin/python scripts/audit_candidate_context.py
.venv/bin/python scripts/sr_distance_prior.py
.venv/bin/python scripts/sr_context.py
.venv/bin/python scripts/summarize_model_validation.py
.venv/bin/python scripts/audit_extended_experiments.py
.venv/bin/python scripts/audit_portable_forest.py
.venv/bin/python -m pytest -q
```

SR置信区间按8个100kb空间块重采样；分类按56个最大窗口重叠组重采样。所有测试位置已被探索性比较观察，不能解释为外部独立验证。大计数缓存和完整重建数组留本地，可由已提交压缩计数分片及训练脚本恢复。

### 固定方案与四候选核验

`scripts/analyze_four_candidates.py`仅分析已固定的U001/U004/U005/U007，输出完整上下文到最近注释的环状距离、两尺度六样本图、中心/上下文强度及共同支持掩码对照；结果在`results/four_candidate_analysis/`。四个候选均不构成已确认新结构类别。U001/U005邻近已知结构；U005的100bp中心有效像素仅约37%，须降低解释置信度。U004中心绝对强度下降不等于相对富集下降；不可将所有条件变化归因于同一机制。

`scripts/freeze_experiment_choices.py`保存分类与SR的操作方案和完整指标对照到`results/experiment_choices/`。分类保留固定三尺度一致性集成，SR以简单计数投影旧CNN作为计数一致性主方案；大感受野、原始CNN及插值均保留。选择是在探索后作出的科学取舍，不是预注册，也不是测试成绩最优声明。报告继续暂停编写。

## 辅助标注工具

运行 `.venv/bin/python scripts/annotation_server.py`，打开 http://127.0.0.1:8765 。支持任意0-based中心坐标、WT两个重复、三尺度矩阵/概率/梯度敏感性、人工确认修正、SQLite持久化与CSV/JSON导出。输入示例U004/U007可直接使用已提交缓存；其他坐标需要原始WT矩阵。工具使用固定折与归一化，明确显示训练/验证重叠限制。来源和完整使用方法见[辅助标注工具](docs/辅助标注工具.md)；数值复核见results/annotation_tool/。这是功能实现与复现核验，尚无人工效率提升实验；报告保持暂停。

## 数据规模与自监督性能对照

当前基准选择保留在`results/experiment_choices/choices.json`，不限制继续改善性能。新计算方案在训练前固定并提交于`results/performance_protocol/plan.json`：SR对比64/352窗口及多尺度对角线外结构损失，架构、960优化步、三种子及空间留出保持一致；分类对比训练区跨重复对比预训练和匹配随机初始化。所有新方案只通过验证选择，现有测试位置仍属于回顾性评价。运行状态见`results/performance_protocol/pipeline_status.json`；正式报告和辅助标注模型保持原样。

```bash
.venv/bin/python scripts/prepare_sr_dense.py
# 从已提交压缩计数分片恢复大数组时：
.venv/bin/python scripts/restore_sr_counts.py
.venv/bin/python scripts/train_sr_data_controls.py
.venv/bin/python scripts/run_performance_followup.py
```

后续脚本顺序执行分类训练、固定SR评价、分类评价、空间组置信区间及完整性检查和图表；任一步失败即停止。远程GPU数据恢复未成功，实际使用本机MPS串行运行，设备记录见results/performance_protocol/hardware.json。未完成的CPU分类产物单独保留，不纳入正式比较。扩展SR训练缓存来自原训练块，验证、测试和WT重复2均未加入SR拟合。

固定对照已完成。SR相对计数投影双三次的PSNR增加0.1505dB、100bp O/E SSIM增加0.00208，8空间块区间均不含零；352窗口本身未提高性能，下游分类不变。跨重复自监督分类没有稳定收益，验证选定策略的Macro-F1和CHID F1反而下降。全部权重、验证决策与逐例错误保留，详见results/performance_validation。不更新辅助工具模型，不继续以当前测试结果调参，正式报告仍暂停。
