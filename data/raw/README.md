完整数据已于 2026-10-13 接入：原有 WT 两重复，加上 NCBI GEO 官方补充矩阵 27 份，共 29 个可读 Cooler、19 个样本条件标签。参考均为 NC_000913.3，10 bp 分箱，长度 4,641,652 bp。压缩原件与解压矩阵均保留，总计约 13.79 GB。

样本清单：data/annotations/samples.csv；逐文件大小、来源、SHA256 和格式核验：results/geo_matrix_audit.json。小体积 NoLig 阴性对照矩阵约 13.7 MB，纳入普通 Git；其余大矩阵本地保存，`.gitignore` 不排除数据。下载器支持续传和单实例锁，解压器采用临时文件完成后原子发布。
