"""Round-eight diagnosis, fair comparisons, conditional decisions and required-task evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.method_review import load_config,write_json
from dlpro.experiment_core import LABELS,LENGTH,ShapeTransform
from dlpro.classification_review import load_inputs,metrics
from summarize_evidence_review import table

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'review';out.mkdir(exist_ok=True);assert not (root/'REPORT.md').exists()
    for stage in ['audit','experiment','explanation','required_evidence']:assert (root/stage/'status.json').exists()
    audit=json.loads((root/'audit/status.json').read_text(encoding='utf-8'));overall=pd.read_csv(root/'experiment/overall.csv');classes=pd.read_csv(root/'experiment/per_class.csv');folds=pd.read_csv(root/'experiment/per_fold.csv');pairs=pd.read_csv(root/'experiment/paired_intervals.csv');case=pd.read_csv(root/'audit/all_baseline_cases_with_quality.csv');hist=pd.read_csv(root/'audit/all_historical_predictions.csv');hp=pd.read_csv(root/'audit/historical_per_class.csv');ho=pd.read_csv(root/'audit/historical_overall.csv');rank=pd.read_csv(root/'audit/baseline_ranking.csv');explain=pd.read_csv(root/'explanation/specificity_intervals.csv');evidence=json.loads((root/'required_evidence/status.json').read_text(encoding='utf-8'));track=pd.read_csv(root/'required_evidence/selected_track_images.csv')
    base_f1=classes[classes.method=='frozen_cnn'].set_index('type').f1;decisions=[]
    for method in ['symmetric_bias','geometric_control','geometric_directional']:
        interval=pairs[(pairs.method==method)&(pairs.reference=='frozen_cnn')&(pairs.metric=='macro_f1')].iloc[0];own=classes[classes.method==method].set_index('type').f1
        merged=folds[folds.method==method].merge(folds[folds.method=='frozen_cnn'],on='fold',suffixes=('_a','_b'));wins=int((merged.macro_f1_a>merged.macro_f1_b).sum());no_harm=bool((own.reindex(base_f1.index)>=base_f1).all());keep=bool(interval.low>0 and no_harm and wins>=3)
        decisions.append(dict(method=method,decision='有条件保留，仍需独立验证' if keep else '不建议替换正式方案',macro_f1_delta=float(interval.delta),low=float(interval.low),high=float(interval.high),folds_improved=wins,folds_total=len(merged),no_class_F1_point_decrease=no_harm,passed_prespecified_retention=keep,formal_default_changed=False))
    write_json(out/'decisions.json',decisions);decision_frame=pd.DataFrame(decisions)
    # Detailed class/space tables; repeated historical methods are descriptive, not independent trials.
    class_fold=[]
    for source,frame in [('historical',hist),('round8',pd.read_csv(root/'experiment/out_of_fold_predictions.csv'))]:
        for (method,fold),part in frame.groupby(['method','fold']):
            y=part.true_type.map(dict(zip(LABELS,range(len(LABELS))))).to_numpy();pred=part.predicted_type.map(dict(zip(LABELS,range(len(LABELS))))).to_numpy();m=metrics(y,pred,len(LABELS))
            for c,label in enumerate(LABELS):class_fold.append(dict(source=source,method=method,fold=fold,type=label,precision=float(m['precision'][c]),recall=float(m['recall'][c]),f1=float(m['f1'][c]),support=int(m['support'][c])))
    pd.DataFrame(class_fold).to_csv(out/'all_methods_fold_class_metrics.csv',index=False)
    recurring=hist.assign(correct=hist.true_type==hist.predicted_type).groupby(['structure_id','true_type','group','fold']).agg(methods=('method','size'),methods_correct=('correct','sum'),prediction_types=('predicted_type',lambda x:','.join(sorted(x.unique())))).reset_index();recurring['all_methods_wrong']=recurring.methods_correct==0;recurring.to_csv(out/'historical_recurring_errors.csv',index=False)
    known=pd.read_csv('data/processed/known_windows.csv');dist=[]
    for i,row in known.iterrows():
        other=known[known.type!=row.type];delta=abs(other.center.to_numpy()-row.center);distance=np.minimum(delta,LENGTH-delta);j=int(np.argmin(distance));near=other.iloc[j]
        dist.append(dict(structure_id=row.structure_id,type=row.type,nearest_other_type=near.type,nearest_other_center_bp=int(distance[j]),overlapping_annotation=bool(row.start<near.end and row.end>near.start),baseline_correct=bool(case.set_index('structure_id').loc[row.structure_id,'correct'])))
    distances=pd.DataFrame(dist);distances.to_csv(out/'other_class_context_diagnostic.csv',index=False)
    distance_summary=distances.groupby(['type','baseline_correct']).agg(cases=('structure_id','size'),nearest_other_median_bp=('nearest_other_center_bp','median'),nearest_other_within500bp=('nearest_other_center_bp',lambda x:int((x<=500).sum())),nearest_annotation_overlaps=('overlapping_annotation','sum')).reset_index()
    qfields=['50bp_rep1_support_fraction','100bp_rep1_mean','100bp_rep1_zero_fraction','100bp_rep1_background_mean','100bp_repeat_log_signal_pearson'];quality=case.groupby(['true_type','correct'])[qfields].median().reset_index();quality.to_csv(out/'quality_medians.csv',index=False)
    hist_compare=ho.merge(hp.pivot(index='method',columns='type',values=['precision','recall','f1']).rename_axis(columns=['metric','type']).pipe(lambda f:f.set_axis([f'{a}_{b}' for a,b in f.columns],axis=1)).reset_index(),on='method')
    hist_compare.to_csv(out/'historical_unified_comparison.csv',index=False)
    # Metric figure displays every new arm, with matched group intervals for the main measures.
    fig,axes=plt.subplots(1,3,figsize=(13,5),constrained_layout=True)
    for ax,metric in zip(axes,['accuracy','macro_f1','f1']):
        part=pairs[(pairs.reference=='frozen_cnn')&(pairs.metric==metric)&(pairs.type==('CHID' if metric=='f1' else 'all'))]
        for i,row in enumerate(part.itertuples()):ax.errorbar(row.delta,i,xerr=[[max(0,row.delta-row.low)],[max(0,row.high-row.delta)]],fmt='o',capsize=3,color='#23698c')
        ax.set_yticks(range(len(part)),part.method);ax.axvline(0,color='gray');ax.set_title(metric+(' (CHID)' if metric=='f1' else ''));ax.set_xlabel('Difference vs frozen CNN; group 95% interval');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Retrospective spatial CV; no independent final-test claim');fig.savefig(out/'classification_paired_effects.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,len(LABELS),figsize=(13,4),constrained_layout=True)
    for ax,label in zip(axes,LABELS):
        sub=explain[explain.type==label]
        for fill,g in sub.groupby('fill'):ax.errorbar(g.bp,g.mean_specificity,yerr=np.vstack([g.mean_specificity-g.low,g.high-g.mean_specificity]),marker='o',capsize=3,label=fill)
        ax.axhline(0,color='gray');ax.set_title(label);ax.set_xlabel('View bin bp (missing points retained as NA)');ax.set_ylabel('True-probability drop: proxy minus background')
    axes[-1].legend(fontsize=7);fig.suptitle('Equal area/distance background control; annotation-square proxy only');fig.savefig(out/'specificity_by_fill.png',dpi=150);plt.close(fig)
    # Deterministic first available example per class, no success filtering.
    f,outer,raw,support=load_inputs();availability=pd.read_csv(root/'explanation/mask_availability.csv');fig,axes=plt.subplots(len(LABELS),3,figsize=(10,10),constrained_layout=True);example_rows=[]
    for r,label in enumerate(LABELS):
        available=availability[(availability.type==label)&(availability.bp==200)&(availability.status=='available')];chosen=available.sort_values('structure_id').iloc[0];i=int(f.index[f.structure_id==chosen.structure_id][0]);fold=int(outer[i]);tf=ShapeTransform.load(f"results/classification_fine/transform_fold{fold}.npz");image=tf.transform(raw[i:i+1])[0,2];saved=np.load(root/'explanation'/f'masks_fold{fold}_200bp.npz');local=int(np.flatnonzero(saved['structure_indices']==i)[0])
        for ax,value,title in zip(axes[r],[image,saved['structure_proxy'][local],saved['background'][local]],['OOF input (WT1, 200bp)','Annotated interval proxy','Equal area/distance background']):ax.imshow(value,origin='lower',cmap='magma');ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        axes[r,0].set_ylabel(str(chosen.structure_id));example_rows.append(dict(structure_id=chosen.structure_id,type=label,bp=200,rule='lexicographically first available; no correctness filtering'))
    fig.savefig(out/'occlusion_mask_examples.png',dpi=135);plt.close(fig);write_json(out/'occlusion_example_rule.json',example_rows)
    track_rows=[dict(requirement='至少三个真实实验条件',teacher='p10',evidence='两套条件分别37C/44C/rif25与37C/HNS_mut/FIS_mut，每条件两个重复；元数据与缓存一致',status='已满足且有验证证据'),dict(requirement='四面板与共享位置轴',teacher='p10',evidence='条件曲线与灰色均值、平均信号、蓝色基因、橙色结构；脚本/协议/代表图一致',status='已满足且有验证证据'),dict(requirement='固定10000bp区间连续覆盖全基因组',teacher='p10',evidence='两套各465条连续索引；0至4641652，末区间截短',status='已满足且有验证证据'),dict(requirement='信号、重复与归一化相符',teacher='p10',evidence=f"全缓存重合并核对；{evidence['task3_raw_signal_checks']}个原始矩阵抽查值吻合，局部±10kb信号、按文库深度归一化",status='已满足且有验证证据'),dict(requirement='图像、注释及内容可读',teacher='p10',evidence=f"按首中末、类别注释最密集、基因最密集、普通背景确定性选{len(track)}张，已实际查看四张联系图；注释计数、图例与位置匹配",status='已满足且有验证证据（定向抽样范围）'),dict(requirement='全部930张逐图视觉检查',teacher='非教师额外硬性要求；完整性边界',evidence='未做；Git对象存在与索引核对不等于逐图视觉检查',status='尚未完成，不冒称完成')]
    pd.DataFrame(track_rows).to_csv(out/'task3_gap.csv',index=False)
    oldsource=json.loads((root/'audit/initial_git_state.json').read_text(encoding='utf-8'));work=pd.read_csv(root/'audit/worklist.csv');cm=pd.read_csv(root/'audit/baseline_confusion.csv',index_col=0);base=overall[overall.method=='frozen_cnn'].iloc[0];new_summary=overall.merge(classes[classes.type=='CHID'][['method','precision','recall','f1']],on='method');credible=[r['method'] for r in decisions if r['passed_prespecified_retention'] and r['method']!='geometric_control'];verdict='达到预先规定的保留门槛，但仍只是回顾性空间验证：'+','.join(credible) if credible else '没有方案达到预先规定的可信替换门槛；保留现有正式CNN'
    current_class=classes[classes.method=='frozen_cnn'];class_counts=pd.crosstab(case.fold,case.true_type).reset_index();errors=case[~case.correct];recurring_summary=recurring.groupby('true_type').agg(cases=('structure_id','size'),all_historical_methods_wrong=('all_methods_wrong','sum'),median_methods_correct=('methods_correct','median')).reset_index()
    for path in (root/'runs').glob('*.json'):
        record=json.loads(path.read_text(encoding='utf-8'))
        if record.get('status')=='failed':write_json(out/(path.stem+'_failure_note.json'),dict(run=path.as_posix(),failure='Strict float32 refit-scale tolerance rejected a maximum 2.3841858e-7 difference before model fitting; tolerance changed to1e-6; frozen transform and checkpoint remained untouched; full OOF probabilities independently matched on retry',source_archive=record['source_archive']))
    directional=overall.set_index('method').loc['geometric_directional'];directional_acc=pairs[(pairs.method=='geometric_directional')&(pairs.reference=='frozen_cnn')&(pairs.metric=='accuracy')].iloc[0];directional_macro=pairs[(pairs.method=='geometric_directional')&(pairs.reference=='frozen_cnn')&(pairs.metric=='macro_f1')].iloc[0]
    body=f'''# 第八轮：分类性能定向优化与必做任务证据补齐

## 最终回答

- **任务一是否可信改善：整体准确率有空间配对支持的提升，但CHID短板未解决，未达到正式替换门槛。** 方向方案Accuracy={directional.accuracy:.6f}，较CNN差{directional_acc.delta:+.6f}，95%区间[{directional_acc.low:.6f},{directional_acc.high:.6f}]；macroF1={directional.macro_f1:.6f}，差值区间[{directional_macro.low:.6f},{directional_macro.high:.6f}]跨零。{verdict}。本轮仅检验两个机制，不再扩展网格或按外层结果调参。
- **CHID主要问题：** 两类形态与决策的混淆集中在CHID/CHIN，且类别少、空间折数量不均；单纯提高CHID召回未解决误报。证据支持这些限制，不能证明唯一因果原因，也不能断言细节不可识别。
- **任务二真实独立证据：没有。** 用户确认暂缺人员，仍{evidence['real_reviewed_cases']}/{evidence['total_cases']}；原公开包和评分冻结文件未变。已交付独立复核操作说明与真实导入/修订/分歧工具检查，临时测试记录已销毁。
- **任务三：核心具体要求已有证据支持满足。** 两套真实三条件、四面板、连续全基因组覆盖；本轮核对全缓存合并与{evidence['task3_raw_signal_checks']}个原始点，定向查看{len(track)}张。没有声称所有930张均已视觉审核。
- **任务五：没有新的证据支持重开大规模搜索。** 复用第七轮强插值、多尺度/守恒取舍结论，未训练或改选超分辨模型。
- **提交硬要求：** 正式报告的设计思路、最终代码ZIP、PPT讲解视频、组号/姓名/学号与贡献百分比、规定命名和分别提交仍需最终确认与制作；本轮没有擅自修改正式成果或推送。

## 1. 第七轮状态与本轮工作清单

实际分支 `{oldsource['branch']}`，HEAD `{oldsource['head']}`。第七轮{audit['seventh_manifest_verified']}条修改/产物清单已逐条复核；本轮开始时记录{audit['protected_files']}个原文件哈希（含用户未提交修改、历史模型、数据缓存及第七轮产物）。最后完整性核验见 `verification/final_status.json`，不能将本轮源码快照当作已提交Git版本。

{table(work)}

原教师PDF已再次直接读取，页码及文本/哈希见 [teacher_requirements.json](audit/teacher_requirements.json)：任务一40、二30、三10为必做；四/五/六选一20，当前选择五。任务一要求准确率/混淆、权重与训练评估脚本、注意力/显著性关注结构；任务三具体四面板在p10；p14评分重论证而非数字。任务二最低标准允许正确复现已知结构，没有强制新簇个数；不据此夸大新颖性结果。

### 修正历史实验数量的解释

实际存在{audit['actual_primary_checkpoint_files']}个主方案模型文件（3种子×5外折）和{audit['actual_primary_transform_files']}个折内归一化文件，**不是已经尝试15种损失权重与5种数据变换网格**。本轮恢复并核对Git小型产物，合计{audit['historical_methods']}个历史方法/种子/选择器预测版本；其中方法高度相关，不能计作独立研究重复。源路径、Git对象及SHA见 [historical_assets.json](audit/historical_assets.json)。没有重训旧CNN、旧平衡采样、核方法或原变换网格。

## 2. 任务一：全部错误与瓶颈

正式CNN复核{audit['n_loci']}条、{audit['n_groups']}空间组：accuracy={base.accuracy:.6f}，macroF1={base.macro_f1:.6f}。

{table(current_class)}

原混淆矩阵（行是真类，列为预测）：

{table(cm.reset_index())}

置信度与错误方向见 [confidence_error_directions.csv](audit/confidence_error_directions.csv)。所有{len(errors)}条错误完整关联原生支持派生的有效像素、密度、零比例、外侧背景、双重复信号一致性；见 [全部样本](audit/all_baseline_cases_with_quality.csv)、[全部错误](audit/all_errors_with_quality.csv)。没有按质量、类别或预测正确性删样。

### 2.1 排序能力与空间不均衡

{table(rank)}

CHID漏检大多落入CHIN；CHID误报也大多来自CHIN。较高的单类排序AUC并不保证稀有类的高精确率，平均精确率与完整PR取舍必须一起看。26个CHID不均匀地分布于外折：

{table(class_counts)}

最小折只有少数CHID，对校准与模型选择影响很大。本轮仍保留原空间组，不以随机逐窗口划分提高分数。

### 2.2 质量、邻近结构与表征

正确/错误样本的合法质量特征中位数（仅描述关联，不当成因果归因）：

{table(quality)}

支持度与粗粒度双重复信号相关性并未在所有误例中明显降低，不能把主要错误简单解释为低质量/无计数。CHIN误判CHID的密度/稀疏度分布与正确样本有差异，说明形态与强度混杂仍需防范；但不能据此证明模型只看强度。

邻近其他标签的几何诊断：

{table(distance_summary)}

注释区间存在包含和邻近，说明中心窗口可能同时呈现多个真实形态，而课程将各行作为互斥三类。全三尺度、双重复的完全相同输入跨标签冲突数为{audit['exact_cross_label_input_pairs']}；**没有据此得到“不可学习”的理论下界**。附近注释、样本少、空间分布与纹理相似只能作为有证据的限制和假设，仍缺独立精细形态标注来区分因果贡献。绝对坐标和注释长度只用于本节诊断，未进入分类器。

### 2.3 过去的方法为何不足

所有历史版本同例汇总：

{table(hist_compare[['method','accuracy','macro_f1','precision_CHID','recall_CHID','f1_CHID','f1_CHIN','f1_OPCID']])}

平衡采样方案提高CHID召回时，仍产生大量CHIN误报，整体macroF1/其他类别有取舍；高整体准确率的核/门控方案也可能牺牲CHID召回。训练多轮、加入细视图或辅助长度目标没有消除该问题，不能把任何单个种子的最高历史分数升级为正式选择。

历史反复出错统计：

{table(recurring_summary)}

逐组表现见 `audit/historical_per_group.csv`，逐折逐类全部对照见 [all_methods_fold_class_metrics.csv](review/all_methods_fold_class_metrics.csv)；重复错误ID与方向见 [historical_recurring_errors.csv](review/historical_recurring_errors.csv)。相关方法不是独立证据，统计只用于定位。

## 3. 两个定向假设与严格空间评估

预先冻结的配置与否证标准见 [PROTOCOL.md](PROTOCOL.md)。所有新方法与当前CNN使用完全相同外层测试位置、原三个训练折、下一循环验证折，最大25.6kb上下文跨角色相交检查通过。旧样本早已被多轮探索过，**本轮是回顾性空间验证，不是新独立测试**。

1. **多类别概率偏置校准：** log(p_c)+b_c，经softmax归一化。所有类别对称使用同一小网格，偏置均值为0；只看内验证折macroF1，依次以偏置范数更小与NLL更小破同分。旧CNN的epoch也在该内验证折选择。本路线采用节制的单内验证划分，未冒称重新训练了完整多内折CNN；外层始终不参与校准。它检验是否主要是决策边界偏移，而非重训旧权重。
2. **方向形态表示：** 训练距离期望下的幅度归一化O/E，提取三物理尺度、三平滑尺度、两距离带的方向Hessian/曲率分位特征，追加到固定几何摘要。没有注释边界、位置、ID或目标类别特例。与只含几何摘要的同一个平衡L2多类逻辑回归配对消融；控制组是该假设的必要基线，不是第三个搜索方向。C只在原三个训练折中做三折空间验证；每内折重拟合ShapeTransform和StandardScaler，随后只用原三个训练折拟合最终模型。

数据角色、拟合索引、内层分数与每外折选择锁全部保存在 `experiment/`。三个种子与两个重复均按位置平均，不当成独立生物样本。保留标准是：macroF1配对95%区间下界>0、每类F1点估计均不下降，且至少3/5外折改善。即便通过也仍需独立验证，本轮不改正式默认。

### 3.1 新旧统一结果

{table(new_summary)}

各类别完整收益/退化：

{table(classes)}

按空间组配对的整体区间：

{table(pairs[(pairs.reference=='frozen_cnn')&pairs.metric.isin(['accuracy','macro_f1'])][['method','metric','delta','low','high','groups']])}

全部逐类precision/recall/F1区间见 [paired_intervals.csv](experiment/paired_intervals.csv)。逐折、逐组、每一错误及混淆矩阵均保留，没有只展示有利方法。

![新旧方案的空间组配对差值](review/classification_paired_effects.png)

### 3.2 预设门槛判定

{table(decision_frame)}

{verdict}。方向方案提升主要体现在CHIN/OPCID，CHID的precision虽略好，recall/F1下降；因此整体Accuracy的真实收益不能替代稀有类完成度。不能只凭单个指标或外折增益宣布成功；也不能以本轮外层结果再次调整偏置/正则参数。方向表示相对几何控制的macroF1配对区间也跨零，尚不能稳健归因于方向特征本身。该轻量分支保留作研究对照，不替换正式CNN，没有启动第三条路线。

## 4. 解释性：结构区代理与公平背景对照

原脚本的中心32×32与四角等面积遮挡并不等于结构/背景标注，而且三个物理视野对应的基因组长度不同。本轮使用独立于模型热图的已有基因组注释区间，在各视图两轴求交形成方形代理。**这是结构区代理，不是接触像素的可靠分割**；真正的分割、结构形态/边界复核仍缺。

只在两个WT重复均有原生支持的非主对角像素定义目标。背景严格匹配每个对角距离的像素数量及总面积，位于代理之外；背景是分散像素，和连续结构方块的遮挡几何仍有差异，属于额外解释限制。无法匹配的长结构/低支持样本保留NA，不删除分类样本。零值、训练OE=1中性值、训练距离中位数三种填充均预先规定且仅训练数据拟合。

真实类别概率下降的结构特异性=结构代理下降−背景下降，正值表示该可比较子集的结构代理更敏感：

{table(explain)}

结果明显依赖类别、视野和填充，CHID最细视野无法构造匹配对照，部分尺度或填充结果反向。因此**不能声称全模型普遍只关注结构**。少数类别/尺度有正向敏感性证据，但不等于因果、精确分割或全类别解释成功。区间只对应可比较子集，不能外推至全部长结构。

真实/错误类别、原预测类和竞争类的逐例变化均见 [per_case.csv](explanation/per_case.csv)，逐类、预测正确性和空间组汇总在同目录。没有用这些遮挡结果选择分类参数。

![三种填充的敏感性与未定义点](review/specificity_by_fill.png)

![按固定规则选出的掩码例图](review/occlusion_mask_examples.png)

## 5. 任务二：真实人员待办，工具与材料保持冻结

用户明确当前没有复核者。本轮没有重新生成原包、没有新增审核标签、没有导入模型判断，仍为{evidence['real_reviewed_cases']}/{evidence['total_cases']}。原ZIP、网页、人工状态和选择锁均在使用前后逐字节核对，保持不变。

[独立复核操作说明](required_evidence/INDEPENDENT_REVIEW_GUIDE.md)可交未参与算法选择者：逐例检查所有分配案例、主动选择判断类别、填写身份/依据/不确定性，独立提交后再讨论。真实独立性需要人员声明；网页默认类别不能当作已审。项目人员只向复核者提供公开包，不能同时提供private、坐标、得分或完整研究报告。真实记录导入前保持待办。

既有导入工具已在一次性临时目录真实执行三次JSON导入，核对两名工程测试身份、修订追加和分歧保留；目录已销毁，不计作真实人员。旧单元测试和本轮真实材料核对各有边界。外部WT文库与矩阵来源映射、补充标签条件和独立性仍未解决，不把旧标签或同源文库包装成独立证据。

## 6. 任务三：具体要求、实际覆盖与限制

{table(pd.DataFrame(track_rows))}

本轮按注释密度、基因密度和边界选图，未用任何模型成功与否筛选。十四张的全部选择规则和路径见 [selected_track_images.csv](required_evidence/selected_track_images.csv)，四张联系图已实际查看；代表图包括两套条件、CHIN/OPCID/CHID密集、普通背景、起点及末端短区间。抽样中四面板、颜色、图例及结构分层排列可读，没有发现需要重画全部图的实现缺陷。结构为空的背景面板不是图片缺失；约570kb示例中的多条件低信号不能解释为某处理的结构消失。

全基因组缓存的条件合并值与样本元数据一致，并从10份原始Cooler核对首/中/末共{evidence['task3_raw_signal_checks']}个位置（含环状起点/末端）。局部±10kb接触信号按文库深度归一化，不能冒称全染色体接触总覆盖。源Git提交、PNG/信号/协议/深度表哈希见 [source_manifest.json](required_evidence/source_manifest.json)；原始点实际误差见 [raw_signal_checks.csv](required_evidence/raw_signal_checks.csv)。Git完整图片集合有930条不等于930张逐图视觉已查。

## 7. 工程、失败记录和复现

入口为 `scripts/run_classification_review.py --stage audit|experiment|explain|evidence|test|report`，配置位于 `configs/classification_review_v1.json`。新目录拒绝覆盖完成阶段，重试使用新的运行attempt；旧失败日志保留。

第一次分类实验在拟合新模型之前因float32变换重算精度检查失败，差值为机器精度量级；重试的容差为1e-6，CNN仍使用原变换，并对每个外层的原概率做独立一致性校验。没有为通过实验改变数据或预期结果。细节见 `runs/experiment.log`、`runs/experiment_precision.log` 和各折 `normalization_refit_check.json`。

每阶段保存真实命令、退出码、Python/库版本、Git HEAD与含未提交代码的源码ZIP；不能把HEAD本身当作全部实验源码。模型、归一化、校准锁、角色清单与预测存于 `experiment/fold*/`。必要测试覆盖组配对指标、空间相交拒绝、等面积/距离/支持掩码、类别对称校准和方向特征反转不变。真实数据阶段另检查外层同例与内层拟合隔离。

本轮最终修改文件清单、原文件哈希核对、源码快照核对、空间划分/掩码检查和运行索引见 `verification/`；正式选择和旧第七轮结果保持不变，没有推送或改正式文档。

## 8. 得分缺口与下一步

1. 分类短板仍应按每类精确率/召回与空间波动解释。是否改进以本轮预设门槛判定，不能按最高历史总分挑模型。若不通过，保留正式CNN，优先补充稀有类和结构区的可信独立标注，停止重复权重网格。
2. 解释性现在有真正按注释区间代理与等面积/距离背景的对照，但可靠像素形态标注仍缺，不能写成结构特异性已完全验证。
3. 任务二缺的是实际独立人员及来源清晰的新证据，而不是更多候选。操作说明与包已可用；安排真实人员前保持待审。
4. 任务三核心要求已有覆盖、数值和代表图依据，可作为必做完成证据；完整图集的最终提交路径/可用性仍应在打包时核对，不需要为本轮重画全部图。
5. 任务五维持冻结。最终课程报告、代码包、PPT视频、成员贡献与命名是提交门槛；本轮仅交付实验研究报告和证据，不替代教师要求的正式三件套。
'''
    (root/'REPORT.md').write_text(body,encoding='utf-8',newline='\n');write_json(out/'status.json',dict(report_completed=True,credible_retained_candidates=credible,formal_choices_changed=False,independent_review_completed=False,task3_core_requirements_supported=True,all_track_images_visually_checked=False,new_model_families=2,no_new_SR_training=True))
    print(verdict,flush=True)

if __name__=='__main__':main()
