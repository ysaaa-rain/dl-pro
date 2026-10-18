# 2026-10-17 GitHub 同步说明

本次依据用户“同步到github”的明确授权，同步此前尚未提交的第五至第八轮项目工作。仓库为 `ysaaa-rain/dl-pro`，目标分支为 `main`；同步前HEAD为 `f44d34892b2786f9804f1d40a5f24b7d9f21b89a`。

## 最新入口

- [第八轮：分类优化、解释性及必做任务证据](../results/classification_review_v1/REPORT.md)
- [第七轮：课程差距、独立证据及强基线](../results/evidence_review_v1/REPORT.md)
- [第六轮：定向训练与对照](../results/targeted_review_v1/review/REPORT.md)
- [第五轮：方法审查](../results/method_review_v1/review_final/REPORT.md)
- [独立复核操作说明](../results/classification_review_v1/required_evidence/INDEPENDENT_REVIEW_GUIDE.md)
- [公开匿名复核目录](../results/evidence_review_v1/blind_review/public/)

## 同步与保留边界

同步代码、配置、测试、实验日志、源码快照、权重、小型中间结果、表格、图片及完整公开复核材料。原正式模型选择不变；任务二仍没有合格新簇、人工审核为0/243。分类方向特征分支有整体准确率收益，但未改善CHID，不替换正式CNN。任务五不重开大规模搜索。

以下文件保留本地，不进入Git：

1. `results/evidence_review_v1/blind_review/private/`：匿名密钥、坐标/算法映射及解盲记录。不能向盲审者提供。
2. `results/evidence_review_v1/blind_review/blind_review_packet.zip`：约170MB，超过GitHub普通仓库100MiB单文件上限，是公开目录的重复打包。全部公开文件仍在仓库；如需单独交审，将public目录打包即可，不附private、算法报告或全仓结果。重新打包的ZIP因时间戳可能不同，其ZIP整体哈希不应冒称与原包相同；公开文件的逐项哈希应一致。
3. 历史协议已声明的大型原始Cooler、原生计数缓存与恢复状态继续保留本地，来源和重建代码保留。没有以轻量检出冒充可直接运行全部原始数据实验。

历史完整性清单可能包含私有文件和本地大ZIP的路径/哈希，这是当时完整本地验收记录，不意味着这些文件已公开，也不是公开克隆的完整自检承诺。公开发布范围以本次同步清单为准；隐私目录和密钥内容不上传。

## 哈希与历史状态

同步前全仓测试为48项通过。第八轮完成时1623个受保护文件核对通过，源码快照、空间划分、852组真实匹配掩码等已有独立记录。

之后为仓库导航和发布边界更新README、进度、忽略规则及Git属性，并清理一个历史新增脚本的行尾空格以通过Git格式检查（无语义变更，前后哈希单列）。这些更新有明确记录；不要将同步后的文件与历史实验开始时的哈希强行比较，也不要改写历史清单来掩盖版本变化。数据、权重、正式选择及既有结果未因此重写，实验原始源码仍在阶段ZIP中保留。

结果目录使用 `-text` 保留归档产物原始字节（部分Windows CSV为CRLF），防止Git自动改换行破坏历史SHA；普通源代码继续使用既有LF规则。历史日志里的Git HEAD、未推送标记与旧绝对路径描述的是当时运行环境；实际实验版本以每阶段源码ZIP、配置及哈希为准。

同步选择与文件哈希清单见 `results/github_sync_20261017/`。正式报告、PPT、讲解视频和最终课程提交包没有在本次同步中重新生成。
