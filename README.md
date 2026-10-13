# 基于深度学习的 Micro-C 接触矩阵实验

当前阶段：GSE272159 官方完整接触矩阵集合已接入，含 29 个矩阵、19 个样本条件标签；344 条源结构注释已提取。首轮真实数据超分辨试验已跑通，CNN 相比双三次插值小幅提升。任务一、二、三及完整任务五仍待推进。

## 环境与运行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test,experiment]'
.venv/bin/python -m pytest -q
.venv/bin/dlpro audit data
.venv/bin/dlpro inspect data/raw/你的样本.cool
.venv/bin/dlpro window data/raw/你的样本.cool 染色体名称 0 10000 --output results/preview
```

最后两条命令需要真实文件与实际染色体名称。默认读取原始计数；`--balance` 明确要求输入已有权重，缺失时报错，不静默替代。窗口最大 2048 bins，避免 10 bp 全基因组稠密化。输出预览为**局部 O/E**，不是正式跨样本归一化结果。跨环状起点窗口暂不支持，明确报错。

## 数据管理

将课程矩阵放入 `data/raw/`，结构坐标放入 `data/annotations/`。`.gitignore` 不排除数据、矩阵、图表或模型权重。小体积数据按普通 Git 提交；单文件达到 50 MiB 先审查，超过 GitHub 普通 Git 限制的文件不能直接提交，采用 LFS 或保留来源和校验清单，不拆分文件规避限制。上传前核实课程数据的共享许可。

课程原始要求 PDF 与补充下载说明保留原位。不得以自动生成的测试样例替代课程数据，也不得把测试通过写成实验完成。

详见 [实验计划](docs/实验计划.md) 和 [进度](docs/进度.md)。

## 超分辨试验

```bash
.venv/bin/python -m dlpro.superresolution --dataset results/sr_pilot --output results/sr_reproduction --epochs 12 --seed 42
```

该命令使用已提交的小体积真实窗口数据复跑，不依赖本地约 907 MiB 的两份原始矩阵。完整矩阵仍保留本地；原始数据来源和 SHA256 见数据清单。训练协议与限制见 [超分辨试验协议](docs/超分辨试验协议.md)。提交前执行 `python scripts/check_staged_sizes.py`，防止误将大矩阵纳入普通 Git。

## 完整矩阵数据（2026-10-13）

在原有两个 WT 基础上，从 NCBI GEO 补齐 27 份矩阵。压缩原件和解压矩阵均位于 `data/raw`，总计约 13.79 GB（12.84 GiB）。所有矩阵均已核验为 NC_000913.3、10 bp 分箱、4,641,652 bp，并完成局部原始计数读取。

- `data/annotations/samples.csv`：样本、条件标签、重复、路径、来源和哈希。
- `results/geo_download_summary.json`：完整性结论与总体积。
- `results/geo_matrix_audit.json`：逐文件格式和校验记录。
- 小体积 NoLig 阴性对照矩阵（约 13.7 MB）纳入普通 Git；其他大矩阵保留本地，数据目录未被忽略。

从远程仓库恢复补充数据：

```bash
python3 scripts/download_geo.py
.venv/bin/python scripts/prepare_geo.py
```

补充下载器恢复 27 个样本；原有 WT 两个矩阵的公开入口见 `results/geo_matrix_audit.json` 的 `source_url`，需下载解压到清单路径。SHA256 是本地计算值；文件大小与官方清单一致，gzip 校验通过，但未取得发布者独立的 SHA256 供对比。
