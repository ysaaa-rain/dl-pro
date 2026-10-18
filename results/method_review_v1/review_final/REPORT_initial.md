# 任务二与任务五：方法范式审查和冻结对照结果

本文件由真实逐例输出自动生成。基线版本f44d348；所有新位置仍属于历史已观察的同一E. coli基因组，未获得外部生物学验证。正式报告、PPT、视频和提交包未改。

## 基线与正确性

见baseline_reproduced/baseline_register.json和reproduction_checks.json。任务二原344条、56空间组、64/128预算形态结果重新推理；任务五六分支和正式投影SubpixelSR检查点重新推理。历史训练未重跑，不把检查点推理复现写成从头训练复现。

修复：旧审计不再写死4个候选/0个簇，改为按数据验证QC、独立性与簇规则。新训练二维网格统计偏差在测试前发现并中止，失败原件和源码留在sr/及runs/train*；正确实验为sr_corrected/。精确训练像素直方图计算归一化，并有完整数组quantile与保留样本隔离检查。

## 任务二

### 数据与根本瓶颈

原始344条全部保留历史复现。新空间留类检验仅使用完整25.6kb输入落在单一完整100kb块的样本。原novelty/运行曾把无折号尾区默认加入训练，评价覆盖登记也相差一条；现novelty_corrected/明确排除无折号输入，保留原运行，作为已观察基因组内的纠错复算。几何边界先固定，尾区缺陷在图审查时发现，不能把纠正运行当全新独立测试。逐项几何信息见data/known_eligibility.csv，最终evaluation_eligible还要求new_fold>=0。越界样本的结果不得由本检验外推。

| type | False | True |
| --- | --- | --- |
| CHID | 8 | 18 |
| CHIN | 78 | 172 |
| OPCID | 15 | 53 |

隐藏CHIN时，剔除共享上下文后已知参考极少，而且没有可与CHIN上下文分离的CHID参考；嵌套标注不是互斥的新形态类别。fits.json记录各折背景、参考、阈值数据的实际索引；diagnostics/novelty_fitting_audit.json核验未使用隐藏上下文或验证/测试折拟合。

### 方法与完整对照

A为多变量条件背景残差及已知参考距离，B为PCA与跨重复浅层自监督表示，C为多尺度/两重复最弱证据联合。所有权重/维数/背景规则均在新评分前固定；无隐藏类型标签参与训练或阈值选择。联合方法已经使用两个重复，其重复相关不称独立验证。

| hidden_type | method | allowed_known_rep1 | background_rep1 | allowed_known_rep2 | background_rep2 |
| --- | --- | --- | --- | --- | --- |
| CHID | conditional_anomaly | 0.59136 | 0.63685 | 0.57963 | 0.65087 |
| CHID | conditional_novelty | 0.64049 | 0.66310 | 0.62469 | 0.67864 |
| CHID | joint_multiscale | 0.55765 | 0.61417 | 0.55765 | 0.61417 |
| CHID | morphology_baseline | 0.68951 | 0.88634 | 0.67667 | 0.89730 |
| CHID | paired_novelty | 0.55247 | 0.57187 | 0.50469 | 0.57467 |
| CHID | pca_novelty | 0.56247 | 0.57543 | 0.51494 | 0.56524 |
| CHIN | conditional_anomaly | 0.38835 | 0.50893 | 0.36476 | 0.52552 |
| CHIN | conditional_novelty | 0.52207 | 0.55537 | 0.53247 | 0.57857 |
| CHIN | joint_multiscale | 0.51208 | 0.54982 | 0.51208 | 0.54982 |
| CHIN | morphology_baseline | 0.42466 | 0.73013 | 0.39912 | 0.74944 |
| CHIN | paired_novelty | 0.49410 | 0.53622 | 0.48207 | 0.55577 |
| CHIN | pca_novelty | 0.48800 | 0.52462 | 0.46286 | 0.54110 |
| OPCID | conditional_anomaly | 0.53570 | 0.55652 | 0.55357 | 0.58975 |
| OPCID | conditional_novelty | 0.52612 | 0.53367 | 0.54335 | 0.56682 |
| OPCID | joint_multiscale | 0.54220 | 0.54328 | 0.54220 | 0.54328 |
| OPCID | morphology_baseline | 0.49930 | 0.76320 | 0.51480 | 0.81201 |
| OPCID | paired_novelty | 0.57175 | 0.57435 | 0.57627 | 0.60325 |
| OPCID | pca_novelty | 0.57443 | 0.57400 | 0.59305 | 0.61165 |

主方案joint相对原形态的已知/留出分离AUROC差及空间组区间：

| hidden_type | repeat | delta | low | high | groups |
| --- | --- | --- | --- | --- | --- |
| CHID | 1 | -0.13185 | -0.32401 | 0.03806 | 45 |
| CHID | 2 | -0.11901 | -0.31350 | 0.04783 | 45 |
| CHIN | 1 | 0.08741 | 0.00680 | 0.16737 | 45 |
| CHIN | 2 | 0.11296 | 0.03003 | 0.19891 | 45 |
| OPCID | 1 | 0.04290 | -0.07577 | 0.16861 | 45 |
| OPCID | 2 | 0.02741 | -0.10523 | 0.15928 | 45 |

全部AUPRC、样本量、背景误报、固定预算逐例召回、配对AUROC区间、跨重复及nuisance相关保存在novelty/和review/的CSV。64为全726格点等效预算，分配到空间合格测试格点后实际选择数另列，不与历史全基因组64候选混用。

### 候选、聚类及取舍

| hidden_type | method | selected | repeat_qc | independent |
| --- | --- | --- | --- | --- |
| CHID | conditional_anomaly | 51 | 33 | 5 |
| CHID | conditional_novelty | 51 | 33 | 7 |
| CHID | joint_multiscale | 51 | 35 | 6 |
| CHID | morphology_baseline | 51 | 35 | 3 |
| CHID | paired_novelty | 51 | 33 | 6 |
| CHID | pca_novelty | 51 | 34 | 6 |
| CHIN | conditional_anomaly | 51 | 34 | 6 |
| CHIN | conditional_novelty | 51 | 29 | 3 |
| CHIN | joint_multiscale | 51 | 34 | 4 |
| CHIN | morphology_baseline | 51 | 39 | 4 |
| CHIN | paired_novelty | 51 | 28 | 3 |
| CHIN | pca_novelty | 51 | 25 | 4 |
| OPCID | conditional_anomaly | 51 | 33 | 8 |
| OPCID | conditional_novelty | 51 | 35 | 8 |
| OPCID | joint_multiscale | 51 | 35 | 7 |
| OPCID | morphology_baseline | 51 | 37 | 3 |
| OPCID | paired_novelty | 51 | 36 | 7 |
| OPCID | pca_novelty | 51 | 35 | 6 |

质检数量增加不能证明新颖性改善。错位重复对照按训练背景密度/稀疏/支持匹配；整体O/E相关和距离层相关分开。所有候选采用同一完整上下文排除和25.6kb独立间隔。未计算全基因组显著性p值；相关窗口和选择偏差不能用朴素多重检验忽略。

| algorithm | non_noise_groups | noise_sets | provisional_shape_clusters |
| --- | --- | --- | --- |
| density_hdbscan | 8 | 72 | 0 |
| fixed_kmeans | 432 | 0 | 0 |

聚类是固定逐窗描述符上的探索性分组，含全部候选及已知上下文，未对测试候选拟合PCA或调节最小成员数。暂定形态簇不等于新生物类别。
固定新颖性替换门槛通过：False。保留原support-aware morphology操作方案；新方法不能仅凭更少误报或更多未注释窗口替换。

## 任务五

### 数据和数学目标

低输入为16个10bp图像像素的均值，输出为非负期望图像计数，块内和等于均值×16。对称Cooler像素不是独立读段。显式模型先预测概率，再乘源质量；零源块概率仍有跨重复监督梯度，但最终输出仍为零。条件目标对原生上三角唯一接触项计数、单独归一化粗对角块，作为依赖块的复合目标。

| split | source_repeat | native_zero_fraction | input_zero_block_fraction | target_mass_in_zero_source_fraction | normalized_within_block_correlation |
| --- | --- | --- | --- | --- | --- |
| train | 1 | 0.87679 | 0.52056 | 0.03454 | 0.04658 |
| train | 2 | 0.87627 | 0.50905 | 0.03050 | 0.04658 |
| val | 1 | 0.86365 | 0.46806 | 0.03006 | 0.04340 |
| val | 2 | 0.86304 | 0.45752 | 0.02660 | 0.04340 |

新训练为21块231窗，稀疏对照42窗；新验证5块10窗、新保留6块12窗，原8测试块16位置继续保留。所有新模型从头训练，两个seed、每臂320步；不能与旧960步模型作纯架构归因。实际感受野5720bp小于完整10240bp输入，全部输入严格落在各自空间块。两个重复是配对观测，不是两个独立基因组。全库深度是固定实验元数据；训练分位数、距离先验与O/E期望仅在新训练位置拟合。

低块内重复相关不是模型可恢复信息的严格理论上限，只说明原生计数监督噪声很大；条件均值仍可能从更多合法数据学到。零源块在严格守恒下不能给另一重复恢复正质量，这是单独的约束冲突。训练量扩展结论限定在相同320步计算预算，不能外推成任何收敛预算下扩充均无效。

### 统一结果

以下为两个seed指标平均，非预测集成；简单基线只算一次。所有方法使用同一输入、目标和评估。50/100bp为非对齐形态指标；160bp是40bp的整数倍，其保真由守恒保证，只作物理对照，不能当作学习收益。

#### new_test / same_repeat

| method | psnr | ssim | oe_50bp_ssim | oe_100bp_ssim | oe_160bp_ssim | raw_nrmse | stabilized_poisson_deviance_per_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| allocation_expanded_cross | 22.28341 | 0.46107 | 0.75384 | 0.92203 | 1.00000 | 0.42053 | 0.69729 |
| allocation_expanded_mixed | 22.29074 | 0.46129 | 0.75433 | 0.92205 | 1.00000 | 0.42051 | 0.69720 |
| bicubic_unprojected | 22.10753 | 0.43570 | 0.73317 | 0.91070 | 0.99205 | 0.42660 | 0.76939 |
| bilinear_unprojected | 21.90101 | 0.42971 | 0.70954 | 0.90578 | 0.98936 | 0.43151 | 0.77077 |
| context_expanded_shape | 22.26636 | 0.46223 | 0.76088 | 0.92223 | 1.00000 | 0.42117 | 0.71790 |
| context_sparse_shape | 22.26222 | 0.46210 | 0.76088 | 0.92214 | 1.00000 | 0.42158 | 0.71876 |
| legacy_sparse_mse | 22.25512 | 0.46156 | 0.75927 | 0.92218 | 1.00000 | 0.42210 | 0.71339 |
| projected_bicubic | 22.18565 | 0.46037 | 0.76069 | 0.92157 | 1.00000 | 0.42402 | 0.73443 |
| projected_bilinear | 22.16217 | 0.46058 | 0.76051 | 0.92224 | 1.00000 | 0.42392 | 0.71053 |
| train_distance_prior | 22.27233 | 0.46092 | 0.75378 | 0.92201 | 1.00000 | 0.42090 | 0.69855 |
| uniform | 21.92341 | 0.45720 | 0.75333 | 0.92168 | 1.00000 | 0.43716 | 0.73104 |

#### new_test / cross_repeat

| method | psnr | ssim | oe_50bp_ssim | oe_100bp_ssim | oe_160bp_ssim | raw_nrmse | stabilized_poisson_deviance_per_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| allocation_expanded_cross | 21.72588 | 0.33008 | 0.20437 | 0.36307 | 0.51334 | 0.44319 | 1.89174 |
| allocation_expanded_mixed | 21.73233 | 0.33022 | 0.20421 | 0.36297 | 0.51334 | 0.44321 | 1.89165 |
| bicubic_unprojected | 21.70373 | 0.32974 | 0.20251 | 0.36379 | 0.52382 | 0.44351 | 1.35405 |
| bilinear_unprojected | 21.59264 | 0.33667 | 0.21011 | 0.37437 | 0.53248 | 0.44495 | 1.15531 |
| context_expanded_shape | 21.70985 | 0.33083 | 0.19891 | 0.35934 | 0.51334 | 0.44380 | 1.91502 |
| context_sparse_shape | 21.70633 | 0.33076 | 0.19899 | 0.35923 | 0.51334 | 0.44421 | 1.91588 |
| legacy_sparse_mse | 21.70203 | 0.33036 | 0.20050 | 0.35957 | 0.51334 | 0.44463 | 1.91503 |
| projected_bicubic | 21.64042 | 0.32941 | 0.19877 | 0.35856 | 0.51334 | 0.44612 | 1.93298 |
| projected_bilinear | 21.61968 | 0.32919 | 0.20080 | 0.35978 | 0.51334 | 0.44605 | 1.90360 |
| train_distance_prior | 21.71670 | 0.32992 | 0.20438 | 0.36310 | 0.51334 | 0.44351 | 1.89296 |
| uniform | 21.40871 | 0.32646 | 0.20407 | 0.36293 | 0.51334 | 0.45842 | 1.92542 |

#### historical_test / same_repeat

| method | psnr | ssim | oe_50bp_ssim | oe_100bp_ssim | oe_160bp_ssim | raw_nrmse | stabilized_poisson_deviance_per_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| allocation_expanded_cross | 21.95796 | 0.44363 | 0.75200 | 0.92406 | 1.00000 | 0.41137 | 0.68832 |
| allocation_expanded_mixed | 21.96396 | 0.44377 | 0.75252 | 0.92408 | 1.00000 | 0.41143 | 0.68839 |
| bicubic_unprojected | 21.77718 | 0.41759 | 0.73256 | 0.91227 | 0.99192 | 0.41746 | 0.75647 |
| bilinear_unprojected | 21.57579 | 0.41029 | 0.70732 | 0.90764 | 0.98943 | 0.42236 | 0.75801 |
| context_expanded_shape | 21.93598 | 0.44423 | 0.75957 | 0.92401 | 1.00000 | 0.41218 | 0.70852 |
| context_sparse_shape | 21.93293 | 0.44412 | 0.75960 | 0.92387 | 1.00000 | 0.41252 | 0.70921 |
| legacy_sparse_mse | 21.92602 | 0.44359 | 0.75807 | 0.92391 | 1.00000 | 0.41300 | 0.70375 |
| projected_bicubic | 21.85622 | 0.44217 | 0.75959 | 0.92313 | 1.00000 | 0.41493 | 0.72425 |
| projected_bilinear | 21.83386 | 0.44246 | 0.75919 | 0.92398 | 1.00000 | 0.41492 | 0.70156 |
| train_distance_prior | 21.94781 | 0.44348 | 0.75193 | 0.92404 | 1.00000 | 0.41179 | 0.68950 |
| uniform | 21.61042 | 0.43974 | 0.75152 | 0.92374 | 1.00000 | 0.42746 | 0.71998 |

#### historical_test / cross_repeat

| method | psnr | ssim | oe_50bp_ssim | oe_100bp_ssim | oe_160bp_ssim | raw_nrmse | stabilized_poisson_deviance_per_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| allocation_expanded_cross | 21.38896 | 0.31952 | 0.21190 | 0.36942 | 0.52901 | 0.43480 | 1.78096 |
| allocation_expanded_mixed | 21.39430 | 0.31963 | 0.21173 | 0.36931 | 0.52901 | 0.43489 | 1.78102 |
| bicubic_unprojected | 21.36433 | 0.31859 | 0.20906 | 0.36885 | 0.53805 | 0.43508 | 1.28397 |
| bilinear_unprojected | 21.25949 | 0.32435 | 0.21671 | 0.38002 | 0.54681 | 0.43647 | 1.09524 |
| context_expanded_shape | 21.36958 | 0.32004 | 0.20618 | 0.36559 | 0.52901 | 0.43558 | 1.80356 |
| context_sparse_shape | 21.36702 | 0.31999 | 0.20630 | 0.36541 | 0.52901 | 0.43590 | 1.80426 |
| legacy_sparse_mse | 21.36251 | 0.31952 | 0.20803 | 0.36569 | 0.52901 | 0.43627 | 1.80257 |
| projected_bicubic | 21.30107 | 0.31848 | 0.20632 | 0.36460 | 0.52901 | 0.43775 | 1.82038 |
| projected_bilinear | 21.28144 | 0.31830 | 0.20849 | 0.36593 | 0.52901 | 0.43775 | 1.79294 |
| train_distance_prior | 21.38071 | 0.31937 | 0.21192 | 0.36947 | 0.52901 | 0.43517 | 1.78210 |
| uniform | 21.08372 | 0.31588 | 0.21174 | 0.36937 | 0.52901 | 0.44945 | 1.81255 |

### 保留决策与未解决问题

测试前统一验证选择：train_distance_prior。主假设mixed相对expanded-context通过预定形态及原生安全门槛：True。验证候选通过新空间与历史块跨重复复核：True。

所有差值和组级bootstrap见sr_corrected/evaluation/paired_intervals.csv；两seed主对照见review/sr_primary_seed_stability.csv。样本扩展比较context_sparse_shape与context_expanded_shape；概率/混合目标比较allocation_expanded_cross与allocation_expanded_mixed，不把架构和训练量同时变化当成单因素收益。

仅在验证选择及冻结测试均满足预定要求时列为推荐候选；否则保留原正式投影首轮CNN。输出稀疏计数、跨重复总量差和同基因组历史暴露限制仍然存在，不能声称恢复了无重复支持的10bp细节或获得外部验证。

## 运行状态、文件与复现

| stage | status | scientific_disposition | exit_code | elapsed_seconds |
| --- | --- | --- | --- | --- |
| baseline | completed | see stage checks and limitations | 0.00000 | 350.69009 |
| candidate_audit | completed | superseded: unassigned partial-tail training inputs | 0.00000 | 42.40657 |
| candidate_audit_corrected | completed | see stage checks and limitations | 0.00000 | 41.47627 |
| cluster_audit | completed | superseded: unassigned partial-tail training inputs | 0.00000 | 62.15657 |
| cluster_audit_corrected | completed | see stage checks and limitations | 0.00000 | 61.90871 |
| diagnose | completed | see stage checks and limitations | 0.00000 | 32.51218 |
| diagnose_corrected | completed | see stage checks and limitations | 0.00000 | 33.21943 |
| evaluate | completed | see stage checks and limitations | 0.00000 | 433.94742 |
| integrity | completed | see stage checks and limitations | 0.00000 | 10.87790 |
| legacy_audit_final | completed | see stage checks and limitations | 0.00000 | 38.01279 |
| novelty | completed | superseded: unassigned partial-tail training inputs | 0.00000 | 795.73196 |
| novelty_corrected | completed | see stage checks and limitations | 0.00000 | 213.09001 |
| novelty_review | completed | superseded: unassigned partial-tail training inputs | 0.00000 | 15.82222 |
| novelty_review_corrected | completed | see stage checks and limitations | 0.00000 | 15.11241 |
| prepare | completed | see stage checks and limitations | 0.00000 | 82.66649 |
| prepare_verify_compatible | failed | see stage checks and limitations | 1.00000 | 5.80935 |
| prepare_verify_compatible_fixed | completed | see stage checks and limitations | 0.00000 | 8.11786 |
| summarize_final | running | see stage checks and limitations | nan | nan |
| test | completed | see stage checks and limitations | 0.00000 | 14.64077 |
| train | failed | superseded: grid quantile bias; stopped before SR test | 1.00000 | 709.27501 |
| train_corrected | completed | see stage checks and limitations | 0.00000 | 2120.42945 |

已实现/已运行/已验证以各stage实际记录、CSV、检查点和测试日志为准。被中止的旧归一化训练是失败/淘汰运行，不混入正确实验。新阶段保存完整源码ZIP、源码/配置/数据清单哈希、设备、命令、起止时间和实际退出状态；早期阶段的源归档清单单列。

运行入口：先配置PYTHONPATH=src，使用项目虚拟环境运行scripts/run_method_review.py。prepare/baseline/novelty使用基础配置；train/evaluate/summarize使用configs/method_review_v1_corrected.json；cluster_audit使用configs/candidate_review_v1.json。每阶段写独立日志，已有记录不会被覆盖；重跑应使用新output目录，且准备数据时也指定对应cache目录。

进一步投入优先级：外部/更深覆盖的匹配原始数据；独立候选的盲法人工形态复核与多条件重复；能区分嵌套类型的新标签任务定义。继续在同一历史测试集搜索更好数字不能替代这些证据。
