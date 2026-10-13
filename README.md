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
