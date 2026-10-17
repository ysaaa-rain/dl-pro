# 第六轮定向性能优化：实际结果

全部位置来自反复观察的同一E. coli基因组。本轮完成实现、实际运行和验证，不把历史空间分组称独立最终测试。原始模型、失败实验、正式选择、报告/PPT/视频/提交包未改，未推送远程。

## 任务二：背景重建是否产生有用新颖性

历史AE已真实训练，不是空类定义。本轮五折权重推理及64/128预算排序复现；首次严格容差被极小MAD放大的float32差异触发，保存失败日志后增加排序一致性核验。历史推理数值误差在historical_ae_reproduction.csv，不冒称从头重训复现。

新方法沿用同一网络，新增支持像素上三角损失、训练背景距离/幅度归一化、训练背景多变量残差校准及严格留类评价。两种子共30模型；重复2只用于评估。背景并不保证无未知结构；隐藏CHIN后独立参考仍很少。残差异常与未知形态并非同一个目标。

### 验证与冻结选择

| hidden_type | fold | background | reference | ae_validation_auc | morphology_validation_auc | joint_validation_auc | fusion_allowed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CHID | 0 | 57 | 74 | 0.75300 | 0.83300 | 0.84600 | False |
| CHID | 1 | 72 | 82 | 0.48833 | 0.63167 | 0.56500 | False |
| CHID | 2 | 66 | 85 | 0.45455 | 0.70034 | 0.61616 | False |
| CHID | 3 | 50 | 93 | 0.34512 | 0.61869 | 0.36027 | False |
| CHID | 4 | 52 | 77 | 0.65000 | 0.64947 | 0.66526 | False |
| CHIN | 0 | 39 | 10 | 0.92500 | 0.95000 | 0.95000 | False |
| CHIN | 1 | 51 | 6 | 0.75000 | 0.87000 | 0.74000 | False |
| CHIN | 2 | 42 | 4 | 0.76389 | 0.87963 | 0.78704 | False |
| CHIN | 3 | 30 | 8 | 0.66667 | 0.83333 | 0.70370 | False |
| CHIN | 4 | 30 | 11 | 0.12000 | 0.86000 | 0.20000 | False |
| OPCID | 0 | 39 | 66 | 0.67500 | 0.78452 | 0.82024 | True |
| OPCID | 1 | 57 | 68 | 0.58095 | 0.66905 | 0.72381 | True |
| OPCID | 2 | 52 | 65 | 0.38657 | 0.82407 | 0.67747 | False |
| OPCID | 3 | 38 | 65 | 0.62257 | 0.82804 | 0.61993 | False |
| OPCID | 4 | 39 | 66 | 0.52609 | 0.77304 | 0.66609 | False |

联合权重固定各半，只有允许参考对背景验证AUROC同时超过AE/形态0.02且信号相关<0.9才启用；失败折回退AE。未按隐藏类别标签选模型。潜变量距离仅是单列诊断，不生成额外候选。

### 全部留类指标（两个方向，含负面结果）

| hidden_type | repeat | method | negative | n_hidden | n_negative | average_precision | hidden_detection | false_positive_rate | auc | low | high | bootstrap_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CHID | 1 | ae_direction_joint | allowed_known | 18 | 225 | 0.11026 | 0.22222 | 0.14667 | 0.61679 | 0.48743 | 0.73472 | 2000 |
| CHID | 1 | ae_direction_joint | background | 18 | 109 | 0.25875 | 0.22222 | 0.07339 | 0.68884 | 0.54598 | 0.81630 | 2000 |
| CHID | 1 | ae_reference_diagnostic | allowed_known | 18 | 225 | 0.08508 | 0.11111 | 0.10667 | 0.54407 | 0.41963 | 0.67032 | 2000 |
| CHID | 1 | ae_reference_diagnostic | background | 18 | 109 | 0.16110 | 0.11111 | 0.08257 | 0.50306 | 0.34181 | 0.66839 | 2000 |
| CHID | 1 | ae_residual | allowed_known | 18 | 225 | 0.11026 | 0.22222 | 0.14667 | 0.61679 | 0.48743 | 0.73472 | 2000 |
| CHID | 1 | ae_residual | background | 18 | 109 | 0.25875 | 0.22222 | 0.07339 | 0.68884 | 0.54598 | 0.81630 | 2000 |
| CHID | 1 | morphology_baseline | allowed_known | 18 | 225 | 0.14008 | 0.55556 | 0.31111 | 0.68951 | 0.58834 | 0.79637 | 2000 |
| CHID | 1 | morphology_baseline | background | 18 | 109 | 0.43910 | 0.55556 | 0.11009 | 0.88634 | 0.80801 | 0.94591 | 2000 |
| CHID | 2 | ae_direction_joint | allowed_known | 18 | 225 | 0.11906 | 0.33333 | 0.16000 | 0.63691 | 0.51172 | 0.75114 | 2000 |
| CHID | 2 | ae_direction_joint | background | 18 | 109 | 0.33399 | 0.33333 | 0.09174 | 0.75306 | 0.62756 | 0.86200 | 2000 |
| CHID | 2 | ae_reference_diagnostic | allowed_known | 18 | 225 | 0.07948 | 0.11111 | 0.12000 | 0.51012 | 0.38910 | 0.63893 | 2000 |
| CHID | 2 | ae_reference_diagnostic | background | 18 | 109 | 0.16026 | 0.11111 | 0.10092 | 0.47273 | 0.31067 | 0.63714 | 2000 |
| CHID | 2 | ae_residual | allowed_known | 18 | 225 | 0.11906 | 0.33333 | 0.16000 | 0.63691 | 0.51172 | 0.75114 | 2000 |
| CHID | 2 | ae_residual | background | 18 | 109 | 0.33399 | 0.33333 | 0.09174 | 0.75306 | 0.62756 | 0.86200 | 2000 |
| CHID | 2 | morphology_baseline | allowed_known | 18 | 225 | 0.13789 | 0.61111 | 0.37333 | 0.67667 | 0.57402 | 0.78602 | 2000 |
| CHID | 2 | morphology_baseline | background | 18 | 109 | 0.47303 | 0.61111 | 0.10092 | 0.89730 | 0.81979 | 0.95486 | 2000 |
| CHIN | 1 | ae_direction_joint | allowed_known | 172 | 71 | 0.64661 | 0.07558 | 0.21127 | 0.43502 | 0.34706 | 0.52034 | 2000 |
| CHIN | 1 | ae_direction_joint | background | 172 | 109 | 0.65458 | 0.07558 | 0.06422 | 0.56014 | 0.46959 | 0.65742 | 2000 |
| CHIN | 1 | ae_reference_diagnostic | allowed_known | 172 | 71 | 0.82509 | 0.11047 | 0.01408 | 0.66738 | 0.58830 | 0.74052 | 2000 |
| CHIN | 1 | ae_reference_diagnostic | background | 172 | 109 | 0.62639 | 0.11047 | 0.07339 | 0.50280 | 0.42706 | 0.58658 | 2000 |
| CHIN | 1 | ae_residual | allowed_known | 172 | 71 | 0.64661 | 0.07558 | 0.21127 | 0.43502 | 0.34706 | 0.52034 | 2000 |
| CHIN | 1 | ae_residual | background | 172 | 109 | 0.65458 | 0.07558 | 0.06422 | 0.56014 | 0.46959 | 0.65742 | 2000 |
| CHIN | 1 | morphology_baseline | allowed_known | 172 | 71 | 0.66733 | 0.32558 | 0.36620 | 0.42466 | 0.35544 | 0.48348 | 2000 |
| CHIN | 1 | morphology_baseline | background | 172 | 109 | 0.77494 | 0.32558 | 0.10092 | 0.73013 | 0.64690 | 0.80249 | 2000 |
| CHIN | 2 | ae_direction_joint | allowed_known | 172 | 71 | 0.63762 | 0.11628 | 0.22535 | 0.40587 | 0.32032 | 0.49012 | 2000 |
| CHIN | 2 | ae_direction_joint | background | 172 | 109 | 0.68381 | 0.11628 | 0.10092 | 0.59542 | 0.50804 | 0.68146 | 2000 |
| CHIN | 2 | ae_reference_diagnostic | allowed_known | 172 | 71 | 0.80549 | 0.08721 | 0.04225 | 0.65992 | 0.57972 | 0.73406 | 2000 |
| CHIN | 2 | ae_reference_diagnostic | background | 172 | 109 | 0.62657 | 0.08721 | 0.07339 | 0.51309 | 0.43653 | 0.59340 | 2000 |
| CHIN | 2 | ae_residual | allowed_known | 172 | 71 | 0.63762 | 0.11628 | 0.22535 | 0.40587 | 0.32032 | 0.49012 | 2000 |
| CHIN | 2 | ae_residual | background | 172 | 109 | 0.68381 | 0.11628 | 0.10092 | 0.59542 | 0.50804 | 0.68146 | 2000 |
| CHIN | 2 | morphology_baseline | allowed_known | 172 | 71 | 0.65687 | 0.40116 | 0.49296 | 0.39912 | 0.33063 | 0.46159 | 2000 |
| CHIN | 2 | morphology_baseline | background | 172 | 109 | 0.79547 | 0.40116 | 0.10092 | 0.74944 | 0.68265 | 0.81245 | 2000 |
| OPCID | 1 | ae_direction_joint | allowed_known | 53 | 190 | 0.34199 | 0.32075 | 0.15789 | 0.57229 | 0.46642 | 0.68198 | 2000 |
| OPCID | 1 | ae_direction_joint | background | 53 | 109 | 0.48101 | 0.32075 | 0.11009 | 0.59365 | 0.46594 | 0.70869 | 2000 |
| OPCID | 1 | ae_reference_diagnostic | allowed_known | 53 | 190 | 0.31212 | 0.20755 | 0.11579 | 0.55070 | 0.45401 | 0.65141 | 2000 |
| OPCID | 1 | ae_reference_diagnostic | background | 53 | 109 | 0.44141 | 0.20755 | 0.08257 | 0.52934 | 0.40745 | 0.64041 | 2000 |
| OPCID | 1 | ae_residual | allowed_known | 53 | 190 | 0.36120 | 0.24528 | 0.05789 | 0.56425 | 0.45625 | 0.67885 | 2000 |
| OPCID | 1 | ae_residual | background | 53 | 109 | 0.45885 | 0.24528 | 0.11009 | 0.56361 | 0.43727 | 0.67980 | 2000 |
| OPCID | 1 | morphology_baseline | allowed_known | 53 | 190 | 0.22770 | 0.37736 | 0.36842 | 0.49930 | 0.40781 | 0.59574 | 2000 |
| OPCID | 1 | morphology_baseline | background | 53 | 109 | 0.56053 | 0.37736 | 0.09174 | 0.76320 | 0.68522 | 0.83544 | 2000 |
| OPCID | 2 | ae_direction_joint | allowed_known | 53 | 190 | 0.38109 | 0.30189 | 0.13684 | 0.62368 | 0.51469 | 0.73114 | 2000 |
| OPCID | 2 | ae_direction_joint | background | 53 | 109 | 0.53834 | 0.30189 | 0.04587 | 0.67907 | 0.55640 | 0.78025 | 2000 |
| OPCID | 2 | ae_reference_diagnostic | allowed_known | 53 | 190 | 0.33276 | 0.24528 | 0.12632 | 0.58749 | 0.48847 | 0.68195 | 2000 |
| OPCID | 2 | ae_reference_diagnostic | background | 53 | 109 | 0.47812 | 0.24528 | 0.09174 | 0.58240 | 0.45971 | 0.69548 | 2000 |
| OPCID | 2 | ae_residual | allowed_known | 53 | 190 | 0.38829 | 0.20755 | 0.03158 | 0.62155 | 0.51274 | 0.72949 | 2000 |
| OPCID | 2 | ae_residual | background | 53 | 109 | 0.49872 | 0.20755 | 0.04587 | 0.65319 | 0.53153 | 0.75517 | 2000 |
| OPCID | 2 | morphology_baseline | allowed_known | 53 | 190 | 0.23609 | 0.43396 | 0.42105 | 0.51480 | 0.41812 | 0.61137 | 2000 |
| OPCID | 2 | morphology_baseline | background | 53 | 109 | 0.61388 | 0.43396 | 0.09174 | 0.81201 | 0.74162 | 0.87681 | 2000 |

背景/已知类型的原始重建误差、校准残差、种子分布见reconstruction_by_category.csv，密度/零值/支持混杂相关见nuisance_correlations.csv。fits.json沿用了第五轮分区元数据；其中旧PCA/paired_training_mse字段不是本轮AE训练结果，当前训练以seed_diagnostics、逐轮曲线和权重为准。

空间配对AUROC差、分类别召回及随机对照详见novelty_vs_morphology.csv、recall_vs_random.csv。64/128是原726点等效预算，实际可评价540点，折内预算按比例分配；不能与旧全基因组预算召回直接混比。

| hidden_type | repeat | method | global_equivalent_budget | center_recall | random_recall | n | delta | low | high | groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CHID | 1 | ae_direction_joint | 64 | 0.26337 | 0.17887 | 243 | 0.08450 | -0.00830 | 0.17426 | 45 |
| CHID | 1 | ae_residual | 64 | 0.26337 | 0.17887 | 243 | 0.08450 | -0.00830 | 0.17426 | 45 |
| CHID | 1 | morphology_baseline | 64 | 0.25103 | 0.17887 | 243 | 0.07216 | -0.02770 | 0.17464 | 45 |
| CHID | 2 | ae_direction_joint | 64 | 0.25926 | 0.17887 | 243 | 0.08039 | -0.00203 | 0.16727 | 45 |
| CHID | 2 | ae_residual | 64 | 0.25926 | 0.17887 | 243 | 0.08039 | -0.00203 | 0.16727 | 45 |
| CHID | 2 | morphology_baseline | 64 | 0.28395 | 0.17887 | 243 | 0.10508 | -0.00303 | 0.22016 | 45 |
| CHIN | 1 | ae_direction_joint | 64 | 0.22222 | 0.17887 | 243 | 0.04335 | -0.03684 | 0.12881 | 45 |
| CHIN | 1 | ae_residual | 64 | 0.22222 | 0.17887 | 243 | 0.04335 | -0.03684 | 0.12881 | 45 |
| CHIN | 1 | morphology_baseline | 64 | 0.25514 | 0.17887 | 243 | 0.07627 | -0.02369 | 0.17809 | 45 |
| CHIN | 2 | ae_direction_joint | 64 | 0.25514 | 0.17887 | 243 | 0.07627 | -0.01567 | 0.18049 | 45 |
| CHIN | 2 | ae_residual | 64 | 0.25514 | 0.17887 | 243 | 0.07627 | -0.01567 | 0.18049 | 45 |
| CHIN | 2 | morphology_baseline | 64 | 0.27572 | 0.17887 | 243 | 0.09685 | -0.01348 | 0.20997 | 45 |
| OPCID | 1 | ae_direction_joint | 64 | 0.23045 | 0.17887 | 243 | 0.05158 | -0.02680 | 0.13445 | 45 |
| OPCID | 1 | ae_residual | 64 | 0.22634 | 0.17887 | 243 | 0.04747 | -0.03404 | 0.13540 | 45 |
| OPCID | 1 | morphology_baseline | 64 | 0.27160 | 0.17887 | 243 | 0.09273 | -0.00904 | 0.20034 | 45 |
| OPCID | 2 | ae_direction_joint | 64 | 0.30453 | 0.17887 | 243 | 0.12565 | 0.02279 | 0.22859 | 45 |
| OPCID | 2 | ae_residual | 64 | 0.26749 | 0.17887 | 243 | 0.08862 | 0.00293 | 0.17897 | 45 |
| OPCID | 2 | morphology_baseline | 64 | 0.29218 | 0.17887 | 243 | 0.11331 | 0.00358 | 0.22871 | 45 |

### 全候选审核

| algorithm | group_instances | provisional_clusters |
| --- | --- | --- |
| density_hdbscan | 32 | 0 |
| fixed_kmeans | 216 | 0 |

| hidden_type | repeat | method | global_equivalent_budget | selected | repeat_qc | unannotated | repeat_qc_unannotated | independent | enough_members_for_any_formal_cluster |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CHID | 1 | ae_direction_joint | 64 | 51 | 47 | 7 | 6 | 6 | True |
| CHID | 1 | ae_direction_joint | 128 | 98 | 69 | 17 | 9 | 9 | True |
| CHID | 1 | ae_residual | 64 | 51 | 47 | 7 | 6 | 6 | True |
| CHID | 1 | ae_residual | 128 | 98 | 69 | 17 | 9 | 9 | True |
| CHID | 1 | morphology_baseline | 64 | 51 | 35 | 5 | 3 | 3 | False |
| CHID | 1 | morphology_baseline | 128 | 98 | 71 | 14 | 10 | 10 | True |
| CHID | 2 | ae_direction_joint | 64 | 51 | 47 | 7 | 5 | 5 | True |
| CHID | 2 | ae_direction_joint | 128 | 98 | 66 | 16 | 8 | 8 | True |
| CHID | 2 | ae_residual | 64 | 51 | 47 | 7 | 5 | 5 | True |
| CHID | 2 | ae_residual | 128 | 98 | 66 | 16 | 8 | 8 | True |
| CHID | 2 | morphology_baseline | 64 | 51 | 37 | 3 | 2 | 2 | False |
| CHID | 2 | morphology_baseline | 128 | 98 | 72 | 12 | 10 | 10 | True |
| CHIN | 1 | ae_direction_joint | 64 | 51 | 45 | 8 | 6 | 6 | True |
| CHIN | 1 | ae_direction_joint | 128 | 98 | 68 | 22 | 11 | 10 | True |
| CHIN | 1 | ae_residual | 64 | 51 | 45 | 8 | 6 | 6 | True |
| CHIN | 1 | ae_residual | 128 | 98 | 68 | 22 | 11 | 10 | True |
| CHIN | 1 | morphology_baseline | 64 | 51 | 39 | 4 | 4 | 4 | False |
| CHIN | 1 | morphology_baseline | 128 | 98 | 70 | 12 | 9 | 9 | True |
| CHIN | 2 | ae_direction_joint | 64 | 51 | 46 | 7 | 5 | 5 | True |
| CHIN | 2 | ae_direction_joint | 128 | 98 | 72 | 22 | 11 | 10 | True |
| CHIN | 2 | ae_residual | 64 | 51 | 46 | 7 | 5 | 5 | True |
| CHIN | 2 | ae_residual | 128 | 98 | 72 | 22 | 11 | 10 | True |
| CHIN | 2 | morphology_baseline | 64 | 51 | 39 | 3 | 3 | 3 | False |
| CHIN | 2 | morphology_baseline | 128 | 98 | 71 | 12 | 9 | 9 | True |
| OPCID | 1 | ae_direction_joint | 64 | 51 | 47 | 8 | 6 | 5 | True |
| OPCID | 1 | ae_direction_joint | 128 | 98 | 75 | 20 | 10 | 9 | True |
| OPCID | 1 | ae_residual | 64 | 51 | 47 | 9 | 6 | 5 | True |
| OPCID | 1 | ae_residual | 128 | 98 | 75 | 21 | 11 | 10 | True |
| OPCID | 1 | morphology_baseline | 64 | 51 | 37 | 4 | 3 | 3 | False |
| OPCID | 1 | morphology_baseline | 128 | 98 | 71 | 13 | 9 | 9 | True |
| OPCID | 2 | ae_direction_joint | 64 | 51 | 47 | 9 | 6 | 6 | True |
| OPCID | 2 | ae_direction_joint | 128 | 98 | 79 | 18 | 8 | 8 | True |
| OPCID | 2 | ae_residual | 64 | 51 | 47 | 9 | 6 | 6 | True |
| OPCID | 2 | ae_residual | 128 | 98 | 80 | 18 | 9 | 9 | True |
| OPCID | 2 | morphology_baseline | 64 | 51 | 36 | 2 | 1 | 1 | False |
| OPCID | 2 | morphology_baseline | 128 | 98 | 70 | 10 | 9 | 9 | True |

覆盖243个唯一候选中心，全部生成统一色标WT热图。簇数为留类/重复/方法/预算的累计实例，不是独立新发现。原最少5个独立成员、已知上下文排除、共同有效像素及0.5重复相关标准未降低。

| ('condition', '') | ('signal_ratio_vs_matched_wt', 'mean') | ('signal_ratio_vs_matched_wt', 'median') | ('signal_ratio_vs_matched_wt', 'count') | ('log_shape_correlation_vs_wt', 'mean') | ('log_shape_correlation_vs_wt', 'median') | ('log_shape_correlation_vs_wt', 'count') | ('log_shape_rmse_vs_wt', 'mean') | ('log_shape_rmse_vs_wt', 'median') | ('log_shape_rmse_vs_wt', 'count') | ('effective_triangle_fraction', 'mean') | ('effective_triangle_fraction', 'median') | ('effective_triangle_fraction', 'count') |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 37C | 1.00000 | 1.00000 | 486 | 0.86176 | 0.85689 | 486 | 0.19442 | 0.19397 | 486 | 0.94747 | 1.00000 | 486 |
| 44C | 0.95985 | 0.94778 | 486 | 0.46510 | 0.44789 | 486 | 0.36639 | 0.36534 | 486 | 0.94747 | 1.00000 | 486 |
| FIS_mut | 0.73716 | 0.71371 | 486 | 0.41559 | 0.42086 | 486 | 0.36240 | 0.35429 | 486 | 0.94747 | 1.00000 | 486 |
| HNS_mut | 0.65808 | 0.63878 | 486 | 0.39105 | 0.39223 | 486 | 0.37304 | 0.37354 | 486 | 0.94747 | 1.00000 | 486 |
| NoLig | 2.85811 | 2.86350 | 243 | 0.04045 | 0.01946 | 243 | 0.86660 | 0.87117 | 243 | 0.94733 | 1.00000 | 243 |
| rif25 | 1.07114 | 1.08179 | 486 | 0.17110 | 0.15602 | 486 | 0.68493 | 0.70277 | 486 | 0.94747 | 1.00000 | 486 |

条件形态和强度对完整列表统一计算，未按有利条件挑候选；NoLig单样本只作描述。完整记录见conditions/，无法恢复的完整原件见conditions/protocol.json。方向联合验证不通过的折没有被静默用作推荐。

留类残差稳健改善门槛：False。当前科学证据不能由已知召回、少量高相关候选或单个好类别替代。保留决策见decisions.json。

## 任务五：双重复自身监督是否有额外贡献

第五轮已经共享使用两个重复，本轮只新增唯一上三角同重复NLL的重复1单源与双重复共享匹配对照；相同初值、空间批次、反向增广、231训练窗/21块、320步、两种子和模型。原形态共享、跨重复NLL、混合目标和简单基线全部复用既有权重与同口径逐例输出，不重跑相同训练。

空目标块无条件分配信息。同重复输入下源零意味着整个原目标块为零；概率仍定义，最终计数严格等于16×块均值。旧conditional_allocation_nll的源正/目标零屏蔽会选中两重复共阳性块，而新唯一接触NLL将源零正目标的潜在概率也纳入跨重复目标；本轮同重复对照避免该源/目标冲突，不声称消除测序噪声。

统一验证选择：train_distance_prior；选择与全部模型哈希在新测试推理前锁定。表中种子指标平均，不是选优种子或预测集成。

| split | comparison | method | psnr | ssim | raw_nrmse | raw_mae | oe_50bp_ssim | oe_100bp_ssim | oe_160bp_ssim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_test | cross_repeat | allocation_expanded_cross | 21.38896 | 0.31952 | 0.43480 | 0.26913 | 0.21190 | 0.36942 | 0.52901 |
| historical_test | cross_repeat | allocation_expanded_mixed | 21.39430 | 0.31963 | 0.43489 | 0.26909 | 0.21173 | 0.36931 | 0.52901 |
| historical_test | cross_repeat | allocation_shared_same | 21.38792 | 0.31949 | 0.43486 | 0.26917 | 0.21190 | 0.36943 | 0.52901 |
| historical_test | cross_repeat | allocation_single_same | 21.38904 | 0.31950 | 0.43481 | 0.26913 | 0.21192 | 0.36942 | 0.52901 |
| historical_test | cross_repeat | bicubic_unprojected | 21.36433 | 0.31859 | 0.43508 | 0.27187 | 0.20906 | 0.36885 | 0.53805 |
| historical_test | cross_repeat | bilinear_unprojected | 21.25949 | 0.32435 | 0.43647 | 0.27396 | 0.21671 | 0.38002 | 0.54681 |
| historical_test | cross_repeat | context_expanded_shape | 21.36958 | 0.32004 | 0.43558 | 0.26912 | 0.20618 | 0.36559 | 0.52901 |
| historical_test | cross_repeat | context_sparse_shape | 21.36702 | 0.31999 | 0.43590 | 0.26924 | 0.20630 | 0.36541 | 0.52901 |
| historical_test | cross_repeat | legacy_sparse_mse | 21.36251 | 0.31952 | 0.43627 | 0.26964 | 0.20803 | 0.36569 | 0.52901 |
| historical_test | cross_repeat | projected_bicubic | 21.30107 | 0.31848 | 0.43775 | 0.27066 | 0.20632 | 0.36460 | 0.52901 |
| historical_test | cross_repeat | projected_bilinear | 21.28144 | 0.31830 | 0.43775 | 0.27130 | 0.20849 | 0.36593 | 0.52901 |
| historical_test | cross_repeat | train_distance_prior | 21.38071 | 0.31937 | 0.43517 | 0.26941 | 0.21192 | 0.36947 | 0.52901 |
| historical_test | cross_repeat | uniform | 21.08372 | 0.31588 | 0.44945 | 0.27760 | 0.21174 | 0.36937 | 0.52901 |
| historical_test | same_repeat | allocation_expanded_cross | 21.95796 | 0.44363 | 0.41137 | 0.25398 | 0.75200 | 0.92406 | 1.00000 |
| historical_test | same_repeat | allocation_expanded_mixed | 21.96396 | 0.44377 | 0.41143 | 0.25393 | 0.75252 | 0.92408 | 1.00000 |
| historical_test | same_repeat | allocation_shared_same | 21.95681 | 0.44359 | 0.41144 | 0.25402 | 0.75197 | 0.92406 | 1.00000 |
| historical_test | same_repeat | allocation_single_same | 21.95822 | 0.44361 | 0.41138 | 0.25397 | 0.75201 | 0.92406 | 1.00000 |
| historical_test | same_repeat | bicubic_unprojected | 21.77718 | 0.41759 | 0.41746 | 0.26056 | 0.73256 | 0.91227 | 0.99192 |
| historical_test | same_repeat | bilinear_unprojected | 21.57579 | 0.41029 | 0.42236 | 0.26491 | 0.70732 | 0.90764 | 0.98943 |
| historical_test | same_repeat | context_expanded_shape | 21.93598 | 0.44423 | 0.41218 | 0.25398 | 0.75957 | 0.92401 | 1.00000 |
| historical_test | same_repeat | context_sparse_shape | 21.93293 | 0.44412 | 0.41252 | 0.25411 | 0.75960 | 0.92387 | 1.00000 |
| historical_test | same_repeat | legacy_sparse_mse | 21.92602 | 0.44359 | 0.41300 | 0.25455 | 0.75807 | 0.92391 | 1.00000 |
| historical_test | same_repeat | projected_bicubic | 21.85622 | 0.44217 | 0.41493 | 0.25577 | 0.75959 | 0.92313 | 1.00000 |
| historical_test | same_repeat | projected_bilinear | 21.83386 | 0.44246 | 0.41492 | 0.25639 | 0.75919 | 0.92398 | 1.00000 |
| historical_test | same_repeat | train_distance_prior | 21.94781 | 0.44348 | 0.41179 | 0.25428 | 0.75193 | 0.92404 | 1.00000 |
| historical_test | same_repeat | uniform | 21.61042 | 0.43974 | 0.42746 | 0.26298 | 0.75152 | 0.92374 | 1.00000 |
| new_test | cross_repeat | allocation_expanded_cross | 21.72588 | 0.33008 | 0.44319 | 0.24441 | 0.20437 | 0.36307 | 0.51334 |
| new_test | cross_repeat | allocation_expanded_mixed | 21.73233 | 0.33022 | 0.44321 | 0.24433 | 0.20421 | 0.36297 | 0.51334 |
| new_test | cross_repeat | allocation_shared_same | 21.72479 | 0.33005 | 0.44322 | 0.24444 | 0.20439 | 0.36307 | 0.51334 |
| new_test | cross_repeat | allocation_single_same | 21.72626 | 0.33007 | 0.44317 | 0.24439 | 0.20440 | 0.36306 | 0.51334 |
| new_test | cross_repeat | bicubic_unprojected | 21.70373 | 0.32974 | 0.44351 | 0.24712 | 0.20251 | 0.36379 | 0.52382 |
| new_test | cross_repeat | bilinear_unprojected | 21.59264 | 0.33667 | 0.44495 | 0.24900 | 0.21011 | 0.37437 | 0.53248 |
| new_test | cross_repeat | context_expanded_shape | 21.70985 | 0.33083 | 0.44380 | 0.24424 | 0.19891 | 0.35934 | 0.51334 |
| new_test | cross_repeat | context_sparse_shape | 21.70633 | 0.33076 | 0.44421 | 0.24439 | 0.19899 | 0.35923 | 0.51334 |
| new_test | cross_repeat | legacy_sparse_mse | 21.70203 | 0.33036 | 0.44463 | 0.24481 | 0.20050 | 0.35957 | 0.51334 |
| new_test | cross_repeat | projected_bicubic | 21.64042 | 0.32941 | 0.44612 | 0.24575 | 0.19877 | 0.35856 | 0.51334 |
| new_test | cross_repeat | projected_bilinear | 21.61968 | 0.32919 | 0.44605 | 0.24636 | 0.20080 | 0.35978 | 0.51334 |
| new_test | cross_repeat | train_distance_prior | 21.71670 | 0.32992 | 0.44351 | 0.24467 | 0.20438 | 0.36310 | 0.51334 |
| new_test | cross_repeat | uniform | 21.40871 | 0.32646 | 0.45842 | 0.25251 | 0.20407 | 0.36293 | 0.51334 |
| new_test | same_repeat | allocation_expanded_cross | 22.28341 | 0.46107 | 0.42053 | 0.23113 | 0.75384 | 0.92203 | 1.00000 |
| new_test | same_repeat | allocation_expanded_mixed | 22.29074 | 0.46129 | 0.42051 | 0.23103 | 0.75433 | 0.92205 | 1.00000 |
| new_test | same_repeat | allocation_shared_same | 22.28223 | 0.46105 | 0.42056 | 0.23115 | 0.75382 | 0.92203 | 1.00000 |
| new_test | same_repeat | allocation_single_same | 22.28403 | 0.46107 | 0.42050 | 0.23111 | 0.75385 | 0.92203 | 1.00000 |
| new_test | same_repeat | bicubic_unprojected | 22.10753 | 0.43570 | 0.42660 | 0.23728 | 0.73317 | 0.91070 | 0.99205 |
| new_test | same_repeat | bilinear_unprojected | 21.90101 | 0.42971 | 0.43151 | 0.24117 | 0.70954 | 0.90578 | 0.98936 |
| new_test | same_repeat | context_expanded_shape | 22.26636 | 0.46223 | 0.42117 | 0.23095 | 0.76088 | 0.92223 | 1.00000 |
| new_test | same_repeat | context_sparse_shape | 22.26222 | 0.46210 | 0.42158 | 0.23110 | 0.76088 | 0.92214 | 1.00000 |
| new_test | same_repeat | legacy_sparse_mse | 22.25512 | 0.46156 | 0.42210 | 0.23159 | 0.75927 | 0.92218 | 1.00000 |
| new_test | same_repeat | projected_bicubic | 22.18565 | 0.46037 | 0.42402 | 0.23268 | 0.76069 | 0.92157 | 1.00000 |
| new_test | same_repeat | projected_bilinear | 22.16217 | 0.46058 | 0.42392 | 0.23328 | 0.76051 | 0.92224 | 1.00000 |
| new_test | same_repeat | train_distance_prior | 22.27233 | 0.46092 | 0.42090 | 0.23140 | 0.75378 | 0.92201 | 1.00000 |
| new_test | same_repeat | uniform | 21.92341 | 0.45720 | 0.43716 | 0.23972 | 0.75333 | 0.92168 | 1.00000 |

### 匹配对照配对空间组区间

| split | comparison | method | reference | metric | delta | low | high | groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_test | cross_repeat | allocation_shared_same | allocation_single_same | psnr | -0.00113 | -0.00185 | -0.00050 | 8 |
| historical_test | cross_repeat | allocation_shared_same | allocation_single_same | ssim | -0.00001 | -0.00002 | 0.00000 | 8 |
| historical_test | cross_repeat | allocation_shared_same | allocation_single_same | raw_nrmse | 0.00005 | 0.00001 | 0.00008 | 8 |
| historical_test | cross_repeat | allocation_shared_same | allocation_single_same | raw_mae | 0.00004 | 0.00003 | 0.00005 | 8 |
| historical_test | cross_repeat | allocation_shared_same | allocation_single_same | oe_50bp_ssim | -0.00001 | -0.00002 | -0.00001 | 8 |
| historical_test | cross_repeat | allocation_shared_same | allocation_single_same | oe_100bp_ssim | 0.00001 | 0.00001 | 0.00002 | 8 |
| historical_test | cross_repeat | allocation_shared_same | allocation_single_same | oe_160bp_ssim | 0.00000 | -0.00000 | 0.00000 | 8 |
| historical_test | same_repeat | allocation_shared_same | allocation_single_same | psnr | -0.00141 | -0.00227 | -0.00067 | 8 |
| historical_test | same_repeat | allocation_shared_same | allocation_single_same | ssim | -0.00001 | -0.00002 | 0.00000 | 8 |
| historical_test | same_repeat | allocation_shared_same | allocation_single_same | raw_nrmse | 0.00005 | 0.00001 | 0.00009 | 8 |
| historical_test | same_repeat | allocation_shared_same | allocation_single_same | raw_mae | 0.00004 | 0.00003 | 0.00006 | 8 |
| historical_test | same_repeat | allocation_shared_same | allocation_single_same | oe_50bp_ssim | -0.00004 | -0.00005 | -0.00003 | 8 |
| historical_test | same_repeat | allocation_shared_same | allocation_single_same | oe_100bp_ssim | -0.00001 | -0.00001 | -0.00000 | 8 |
| historical_test | same_repeat | allocation_shared_same | allocation_single_same | oe_160bp_ssim | -0.00000 | -0.00000 | 0.00000 | 8 |
| new_test | cross_repeat | allocation_shared_same | allocation_single_same | psnr | -0.00147 | -0.00208 | -0.00087 | 6 |
| new_test | cross_repeat | allocation_shared_same | allocation_single_same | ssim | -0.00002 | -0.00003 | -0.00001 | 6 |
| new_test | cross_repeat | allocation_shared_same | allocation_single_same | raw_nrmse | 0.00005 | 0.00002 | 0.00008 | 6 |
| new_test | cross_repeat | allocation_shared_same | allocation_single_same | raw_mae | 0.00004 | 0.00003 | 0.00006 | 6 |
| new_test | cross_repeat | allocation_shared_same | allocation_single_same | oe_50bp_ssim | -0.00002 | -0.00002 | -0.00001 | 6 |
| new_test | cross_repeat | allocation_shared_same | allocation_single_same | oe_100bp_ssim | 0.00001 | 0.00001 | 0.00001 | 6 |
| new_test | cross_repeat | allocation_shared_same | allocation_single_same | oe_160bp_ssim | 0.00000 | -0.00000 | 0.00000 | 6 |
| new_test | same_repeat | allocation_shared_same | allocation_single_same | psnr | -0.00180 | -0.00248 | -0.00111 | 6 |
| new_test | same_repeat | allocation_shared_same | allocation_single_same | ssim | -0.00002 | -0.00003 | -0.00001 | 6 |
| new_test | same_repeat | allocation_shared_same | allocation_single_same | raw_nrmse | 0.00006 | 0.00003 | 0.00010 | 6 |
| new_test | same_repeat | allocation_shared_same | allocation_single_same | raw_mae | 0.00005 | 0.00004 | 0.00006 | 6 |
| new_test | same_repeat | allocation_shared_same | allocation_single_same | oe_50bp_ssim | -0.00003 | -0.00004 | -0.00003 | 6 |
| new_test | same_repeat | allocation_shared_same | allocation_single_same | oe_100bp_ssim | -0.00001 | -0.00001 | -0.00000 | 6 |
| new_test | same_repeat | allocation_shared_same | allocation_single_same | oe_160bp_ssim | 0.00000 | -0.00000 | 0.00000 | 6 |

新空间组跨重复100bp差+0.000010，95%组区间[+0.000006,+0.000015]；历史块差+0.000013。新组同重复50bp差-0.000033。必须同时考察PSNR/SSIM和对训练先验的比较，不能把单个形态数字的提高等同细节恢复。

### 下游已知结构与噪声诊断

19个实际完整的已知注释进入局部形态/信号峰位诊断，全部方法均评价。known_structure_per_window.csv保留逐例结果，known_structure_paired_intervals.csv列出空间组配对区间。10.24kb输入不足以运行要求完整12.8/25.6kb上下文的任务一分类器；本轮不补零冒充分类准确率，原49个全上下文下游基线仍保留。

known_examples/按基因组坐标固定选取每类第一个合格实例，图中青线是原注释边界，并非模型预测边界；不按模型改善选图。另保留两个固定随机空间探针。可用scripts/reconstruct_shared_same.py --input-low <40bp均值.npy> --output <新输出.npz>直接应用验证选择的分支，或用--method allocation_shared_same应用本轮网络对照；应用示例与冻结推理逐像素核对。

噪声表保留全部241训练/验证窗、全部距离/计数层（含空层）；每个原生唯一40bp块的双重复总量、分配MSE/余弦值保存在可重建本地缓存及哈希清单。20个验证方向的直接Cooler聚合与native均值检查通过。跨重复相关只是经验诊断，不是理论信息上限。

下表相关系数为各窗口相应层内相关的算术平均，未定义者不伪填0，并同时列出有效窗口数与全部块数；不将不同层/不同样本量的相关混作全局相关。

| distance_lower_bp | distance_upper_bp | count_lower | count_upper | total_blocks | both_positive_blocks | mean_window_input40_pearson | defined_input40_correlations | mean_window_common_positive_detail_pearson | defined_detail_correlations | mean_window_proportion_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 200 | 0 | 1 | 8706 | 0 | -0.34616 | 110 | nan | 0 | 0.05069 |
| 0 | 200 | 1 | 4 | 13902 | 9747 | -0.52590 | 237 | 0.19846 | 240 | 0.09157 |
| 0 | 200 | 4 | 16 | 40088 | 39486 | 0.11092 | 241 | 0.35922 | 241 | 0.04221 |
| 0 | 200 | 16 | 64 | 63670 | 63670 | 0.75710 | 241 | 0.61578 | 241 | 0.00386 |
| 0 | 200 | 64 | 256 | 178683 | 178683 | 0.84113 | 241 | 0.39471 | 241 | 0.00089 |
| 0 | 200 | 256 | 1000000000 | 1021 | 1021 | 0.33962 | 18 | 0.52040 | 26 | 0.00042 |
| 200 | 1000 | 0 | 1 | 188655 | 0 | -0.42187 | 240 | nan | 0 | 0.03610 |
| 200 | 1000 | 1 | 4 | 349420 | 257956 | -0.30461 | 241 | 0.03924 | 241 | 0.06360 |
| 200 | 1000 | 4 | 16 | 239784 | 238750 | 0.36927 | 241 | 0.10803 | 241 | 0.02390 |
| 200 | 1000 | 16 | 64 | 189645 | 189645 | 0.81186 | 241 | 0.31060 | 241 | 0.00442 |
| 200 | 1000 | 64 | 256 | 196079 | 196079 | 0.86810 | 241 | 0.37942 | 241 | 0.00111 |
| 200 | 1000 | 256 | 1000000000 | 447 | 447 | 0.24087 | 9 | 0.51518 | 16 | 0.00043 |
| 1000 | 100000 | 0 | 1 | 4258636 | 0 | -0.25913 | 241 | nan | 0 | 0.02514 |
| 1000 | 100000 | 1 | 4 | 1958922 | 1231102 | -0.44495 | 241 | 0.03164 | 241 | 0.06878 |
| 1000 | 100000 | 4 | 16 | 223691 | 222044 | -0.34822 | 223 | 0.08925 | 234 | 0.03489 |
| 1000 | 100000 | 16 | 64 | 16545 | 16545 | 0.00366 | 38 | 0.32227 | 57 | 0.00718 |
| 1000 | 100000 | 64 | 256 | 42 | 42 | -0.07789 | 4 | 0.62662 | 4 | 0.00149 |
| 1000 | 100000 | 256 | 1000000000 | 0 | 0 | nan | 0 | nan | 0 | nan |

repeat_error_decomposition.csv验证精确代数恒等式：两个重复目标的平均MSE=对两重复均值的MSE+两重复差平方/4。后项含生物差异和采样误差，两个重复不足以把它们分离，也不足以断言不可预测噪声的严格比例。

### 最终保留与停止条件

共享训练相对匹配单源的跨重复100bp组区间在两套历史空间组均为正：True。验证选中方案为train_distance_prior，但是否替换正式默认还需要全部关键指标和局部结构取舍。原choices.json保持冻结；本轮停止网络结构/损失搜索，不以同一历史测试结果继续改权重。

## 实现、运行、验证与待完成

已实现并运行：旧AE权重复现、支持掩码背景AE/固定方向联合/随机对照、留类新颖性、全部候选跨重复/聚类/条件核验、四个匹配SR模型、全窗口噪声诊断、统一验证与同/跨双方向评价。

已验证：原有工作哈希不变、来源归档哈希、相同分区与旧形态基线复现、原始聚合数学语义、新SR计数/对称/非负、全候选覆盖。单元测试及真实退出状态见runs/与run_status.csv。AE工程加速中止保留原模型后重读，并按所需索引逐值核验；没有按测试成绩改变方法。

待完成科学证据：外部/更深覆盖匹配重复、独立盲法候选复核、能区分嵌套结构类型的新颖性任务定义。本轮实验已经收尾，不代表新生物类型或可靠10bp细节恢复已解决。

运行入口：设置PYTHONPATH=src，使用项目Python运行scripts/run_targeted_review.py --stage <阶段>。配置、原数据缓存及第五轮比较权重为依赖；日志不可覆盖，复跑需新output目录。noise/conditions的大缓存本地保留，来源及SHA256已记录，可按脚本恢复。

修改文件清单见change_manifest.csv，运行状态见run_status.csv；完整CSV保留不利指标及淘汰理由。下一步优先投入外部数据和标签核验，不再以增加同类实验数量替代新证据。