"""Evidence-bounded course audit and frozen-model decisions, computed from real artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.method_review import load_config,write_json
from dlpro.experiment_core import LENGTH

def table(frame):
    def cell(value):
        if isinstance(value,(float,np.floating)):return '' if pd.isna(value) else f'{value:.6g}'
        return str(value).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |',*['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])

def paired_auc(labels,first,second,groups,cfg):
    labels=np.asarray(labels,bool);unique,inverse=np.unique(groups,return_inverse=True);p=np.flatnonzero(labels);n=np.flatnonzero(~labels)
    wins=np.zeros((len(unique),len(unique)));differences=[]
    for scores in [np.asarray(first),np.asarray(second)]:differences.append((scores[p,None]>scores[n]).astype(float)+.5*(scores[p,None]==scores[n]))
    delta=differences[0]-differences[1];np.add.at(wins,(inverse[p,None],inverse[n][None]),delta)
    pos=np.bincount(inverse[p],minlength=len(unique));neg=np.bincount(inverse[n],minlength=len(unique))
    weights=np.random.default_rng(cfg['seed']).multinomial(len(unique),np.full(len(unique),1/len(unique)),cfg['bootstrap_draws']);den=(weights@pos)*(weights@neg)
    samples=np.einsum('bi,ij,bj->b',weights,wins,weights)[den>0]/den[den>0];low,high=np.quantile(samples,[.025,.975])
    return dict(delta=float(delta.mean()),low=float(low),high=float(high),groups=len(unique),positive=len(p),negative=len(n),bootstrap_valid=len(samples))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();cfg=load_config(args.config)
    root=Path(cfg['output']);out=root/'review';out.mkdir(exist_ok=True)
    if (root/'REPORT.md').exists():
        versions=out/'report_versions';versions.mkdir(exist_ok=True);backup=versions/f'revision_{len(list(versions.iterdir()))+1:02d}';backup.mkdir()
        shutil.copy2(root/'REPORT.md',backup/'REPORT.md')
        for existing in out.iterdir():
            if existing.is_file():shutil.copy2(existing,backup/existing.name)
    assert (root/'sr_audit/status.json').exists() and (root/'material_verification/status.json').exists()
    read=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
    original=Path(cfg['sixth_output']);fits=json.loads((original/'background_ae/fits.json').read_text(encoding='utf-8'))
    summary=pd.read_csv(original/'background_ae/novelty_summary.csv');folds=pd.read_csv(root/'novelty_diagnosis/per_fold_novelty.csv');scores=pd.read_csv(root/'novelty_diagnosis/all_scores_with_nuisance.csv')
    # Every comparison uses identical cases and group bootstrap draws.
    auc_rows=[]
    for (hidden,repeat),part in scores.groupby(['hidden_type','repeat']):
        for negative in ['allowed_known','background']:
            part_use=part[(part.actual_type==hidden)|((part.actual_type=='background') if negative=='background' else ((part.actual_type!='background')&(part.actual_type!=hidden)))]
            baseline=part_use[part_use.method=='morphology_baseline']
            for method in ['ae_residual','ae_direction_joint','ae_reference_diagnostic']:
                pairs=part_use[part_use.method==method].merge(baseline,on=['item_id','group','actual_type'],suffixes=('_a','_b'),validate='one_to_one')
                auc_rows.append(dict(hidden_type=hidden,repeat=repeat,negative_set=negative,method=method,reference='morphology_baseline',**paired_auc(pairs.actual_type.eq(hidden),pairs.score_a,pairs.score_b,pairs.group,cfg)))
    auc=pd.DataFrame(auc_rows);auc.to_csv(out/'paired_auc_differences.csv',index=False)
    # Distribution counts are unique cases, rather than the four repeated score methods.
    unique=scores.drop_duplicates(['hidden_type','fold','repeat','item_id'])
    nuis=unique.groupby(['hidden_type','repeat','actual_type']).agg(cases=('item_id','size'),log_density_median=('log_density','median'),zero_fraction_median=('zero_fraction','median'),support_fraction_median=('support_fraction','median'),valid_unique_pixels_median=('valid_unique_pixels','median')).reset_index();nuis.to_csv(out/'unique_class_nuisance.csv',index=False)
    corr=pd.read_csv(root/'novelty_diagnosis/within_class_nuisance_correlations.csv')
    corr_summary=corr.groupby(['hidden_type','method','actual_type','nuisance']).agg(finite_folds=('spearman','count'),median_spearman=('spearman','median'),max_absolute_spearman=('spearman',lambda x:float(x.abs().max()) if x.notna().any() else np.nan),minimum_case_count=('n','min'),maximum_case_count=('n','max')).reset_index();corr_summary.to_csv(out/'within_class_correlation_summary.csv',index=False)
    fold_range=folds[(folds.repeat==1)&(folds.negative=='allowed_known')].groupby(['hidden_type','method']).agg(folds=('fold','size'),available=('auc','count'),hidden_min=('n_hidden','min'),hidden_max=('n_hidden','max'),auc_min=('auc','min'),auc_max=('auc','max')).reset_index();fold_range.to_csv(out/'fold_range.csv',index=False)
    fit_rows=[dict(hidden_type=f['hidden_type'],fold=f['fold'],background_training=len(f['background_training_indices']),known_references=len(f['reference_indices']),validation_references=len(f['calibration_reference_indices']),reference_groups=f['reference_groups'],fusion_enabled=f['joint_allowed'],validation_signal_spearman=f['validation_signal_spearman'],**{'validation_'+k:v for k,v in f['validation_auc'].items()}) for f in fits]
    fit_table=pd.DataFrame(fit_rows);fit_table.to_csv(out/'fit_sample_counts_and_gates.csv',index=False)
    # Supplementary workbooks are audited as labels with unresolved condition scope.
    known=pd.read_csv('data/processed/known_windows.csv');label_rows=[]
    for name in ['Hp.xlsx','Hp newly emerged.xlsx']:
        source=root/'independent_data'/name;book=pd.read_excel(source)
        for row in book.itertuples(index=False,name=None):
            values=dict(zip(book.columns,row));start=int(values['Start']);end=int(values['End']);hit=known[(known.start<end)&(known.end>start)];nearest=(abs(known[known.type=='CHIN'].start-start)+abs(known[known.type=='CHIN'].end-end)).min()
            label_rows.append(dict(workbook=name,start=start,end=end,chrom=str(values['Chr']),overlapping_known=len(hit),overlap_types=','.join(sorted(hit.type.unique())),minimum_CHIN_endpoint_L1_bp=int(nearest),within_genome=0<=start<end<=LENGTH,condition_scope='not established from workbook alone',independent_labels=False,used_for_algorithm=False))
    labels=pd.DataFrame(label_rows);labels.to_csv(out/'supplemental_coordinate_compatibility.csv',index=False)
    # Full baseline tables and paired effects remain native protocol metrics.
    all_metrics=pd.read_csv(root/'sr_audit/unified_all_metrics.csv');effects=pd.read_csv(root/'sr_audit/all_paired_intervals.csv');strata=pd.read_csv(root/'sr_audit/stratified_paired_intervals.csv');strat_summary=pd.read_csv(root/'sr_audit/stratified_summary.csv')
    example=effects[(effects.split=='new_test')&(effects.comparison=='cross_repeat')&(effects.method=='train_distance_prior')&(effects.reference=='bicubic_unprojected')];example.to_csv(out/'prior_vs_raw_bicubic.csv',index=False)
    improved=[]
    for key,part in strata.groupby(['split','comparison','axis','method','reference','metric']):
        improved.append(dict(zip(['split','comparison','axis','method','reference','metric'],key),defined_cells=int(part.delta.notna().sum()),empty_or_undefined=int(part.delta.isna().sum()),point_estimate_lower=int((part.delta<0).sum()),interval_entirely_lower=int((part.high<0).sum()),interval_entirely_higher=int((part.low>0).sum())))
    improvements=pd.DataFrame(improved);improvements.to_csv(out/'stratum_effect_summary.csv',index=False)
    keys=['split','comparison','axis','distance_lower_bp','distance_upper_bp','layer_lower','layer_upper']
    baseline=strat_summary[strat_summary.method=='train_distance_prior'].melt(id_vars=keys,value_vars=['raw_mae','raw_mse','log_mse','conditional_proportion_mse'],var_name='metric',value_name='reference_error')
    relative=strata[strata.reference=='train_distance_prior'].merge(baseline,on=keys+['metric'],validate='many_to_one');relative['descriptive_percent_reduction']=-100*relative.delta/relative.reference_error.where(relative.reference_error>0);relative.to_csv(out/'relative_stratum_effects.csv',index=False)
    sw=pd.read_csv(root/'sr_audit/stratified_per_window.csv');composition=sw[(sw.method=='train_distance_prior')&(sw.comparison=='cross_repeat')].groupby(['split','axis','distance_lower_bp','distance_upper_bp','layer_lower','layer_upper']).agg(blocks=('blocks','sum'),unique_pixels=('unique_pixels','sum'),empty_window_directions=('blocks',lambda x:int((x==0).sum())),window_directions=('blocks','size'),common_positive_blocks=('common_positive_blocks','sum')).reset_index();composition['pixel_fraction_within_axis']=composition.unique_pixels/composition.groupby(['split','axis']).unique_pixels.transform('sum');composition.to_csv(out/'stratum_composition.csv',index=False)
    noise=pd.read_csv(root/'sr_audit/reused_train_val_noise.csv');noise_summary=[]
    for key,part in noise.groupby(['distance_lower_bp','distance_upper_bp','count_lower','count_upper']):
        item=dict(zip(['distance_lower_bp','distance_upper_bp','count_lower','count_upper'],key),window_cells=len(part),empty_cells=int((part.blocks==0).sum()),blocks=int(part.blocks.sum()),both_positive_blocks=int(part.both_positive_blocks.sum()))
        for name in ['input40_pearson','normalized_detail_pearson_common_positive','source1_zero_block_fraction','source2_zero_block_fraction','native1_zero_fraction','native2_zero_fraction']:item[name+'_median']=float(part[name].median()) if part[name].notna().any() else None
        noise_summary.append(item)
    noise_summary=pd.DataFrame(noise_summary);noise_summary.to_csv(out/'reused_noise_summary_all_cells.csv',index=False)
    # Fixed plots display every baseline and every configured stratum, including NA cells.
    names=sorted(all_metrics.method.unique());fig,axes=plt.subplots(1,4,figsize=(17,7),sharey=True,constrained_layout=True)
    for ax,metric in zip(axes,['psnr','oe_50bp_ssim','oe_100bp_ssim','oe_160bp_ssim']):
        use=effects[(effects.split=='new_test')&(effects.comparison=='cross_repeat')&(effects.reference=='bicubic_unprojected')&(effects.metric==metric)].set_index('method')
        for i,name in enumerate(names):
            if name=='bicubic_unprojected':ax.plot(0,i,'o',color='black');continue
            row=use.loc[name];ax.errorbar(row.delta,i,xerr=[[max(0,row.delta-row.low)],[max(0,row.high-row.delta)]],fmt='o',color='#276b91',capsize=2)
        ax.axvline(0,color='gray',linewidth=1);ax.set_title(metric);ax.set_xlabel('Difference vs raw bicubic; group 95% interval');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(names)),names);fig.suptitle('Frozen models: new spatial groups, cross repeat (6 groups; retrospective)');fig.savefig(out/'strong_baseline_intervals.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for column,axis in enumerate(['count','effective_pixels']):
        for row_index,metric in enumerate(['raw_mse','conditional_proportion_mse']):
            use=relative[(relative.split=='new_test')&(relative.comparison=='cross_repeat')&(relative.axis==axis)&(relative.method=='allocation_shared_same')&(relative.metric==metric)].sort_values(['layer_lower','distance_lower_bp'])
            pivot=use.pivot(index='layer_lower',columns='distance_lower_bp',values='descriptive_percent_reduction');array=pivot.to_numpy();vmax=max(float(np.nanmax(abs(array))),1e-6);ax=axes[row_index,column]
            cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#dddddd');im=ax.imshow(np.ma.masked_invalid(array),cmap=cmap,vmin=-vmax,vmax=vmax,aspect='auto')
            for i in range(len(pivot)):
                for j in range(len(pivot.columns)):ax.text(j,i,'NA' if np.isnan(array[i,j]) else f'{array[i,j]:.2f}',ha='center',va='center',fontsize=9)
            ranges=use[['layer_lower','layer_upper']].drop_duplicates().sort_values('layer_lower');ax.set_yticks(range(len(pivot)),[f'[{x.layer_lower:g}, {x.layer_upper:g})' for x in ranges.itertuples()]);ax.set_xticks(range(len(pivot.columns)),['0-200','200-1000','1000+']);ax.set_title(f'{axis}: {metric}');ax.set_xlabel('Distance bp');fig.colorbar(im,ax=ax,label='% error reduction vs distance prior')
    fig.suptitle('Shared same-repeat model: full frozen strata; NA retained; descriptive percentages');fig.savefig(out/'all_strata_effects.png',dpi=150);plt.close(fig)
    frozen=read('course_audit/frozen_choices.json');classification=read('course_audit/classification_verified.json');geo=read('independent_data/compatibility_summary.json');blind=read('blind_review/automatic_quality_audit.json');human=read('blind_review/human_review_status.json');sr_status=read('sr_audit/status.json');geometry=pd.read_csv(root/'novelty_diagnosis/candidate_context_taxonomy.csv');gaps=pd.read_csv(root/'course_audit/grading_gap.csv')
    decisions=[
        dict(item='支持区域感知形态评分与既定候选规则',decision='保留并推荐',evidence='作为现有可复现候选筛选流程；不认定新簇',cost='复用冻结模型与预算',risk='仍无独立新颖性证据'),
        dict(item='完整匿名复核包与追加式复核工具',decision='保留并推荐',evidence=f"{blind['unique_cases']}例完整生成，真实浏览器检查通过；人工已审{human['reviewed']}例",cost='需真实人员逐例审阅',risk='工程可用不等于独立审核完成'),
        dict(item='背景自编码器残差/方向联合替换正式发现方案',decision='不建议采用',evidence='各类别/空间折不稳定，联合门控多数回退，正式新簇仍为0',cost='本轮不重新训练',risk='异常重建误差不是新颖性证明'),
        dict(item='AE到已知参考的距离诊断',decision='有条件保留，仍需验证',evidence='仅部分隐藏类别出现正信号；只做误差诊断，不生成候选',cost='复用编码与参考',risk='类别形态差异和参考数量混杂；不可测试调权'),
        dict(item='将嵌套/邻近形态列为单独待审类别',decision='有条件保留，仍需验证',evidence='原标签确有重叠；本轮只生成几何关系，不更改正式排除',cost='需要生物学判据与标签',risk='几何相交不能证明新类'),
        dict(item='外部WT t1/t2数据与补充工作簿作为独立真值',decision='当前证据不足，暂不决策',evidence='文库到已用矩阵的合并关系及新增标签条件未确认；三类BED未更新',cost='先核实来源，再决定是否获取大型数据',risk='不能重复计数或把跨条件注释当WT真值'),
        dict(item='train_distance_prior替换正式超分辨默认',decision='不建议采用',evidence='相对较强原始插值的100/160bp形态退化，SSIM提升不稳；未达到一致实质优势',cost='轻量先验仍可作对照',risk='只按某个验证指标切换会掩盖多指标取舍'),
        dict(item='train_distance_prior作为严格守恒分支基线',decision='有条件保留，仍需验证',evidence='满足非负、对称、聚合；部分原始误差与50bp改善',cost='无需训练新网络',risk='不宣称恢复独立10bp真值，160bp输出非学习所得'),
        dict(item='双重复共享网络继续增加结构/损失搜索',decision='不建议采用',evidence='相对匹配单重复收益极小；相对强基线不一致；计数/尺度冲突无容量不足证据',cost='停止本数据下继续网格试验',risk='继续调历史测试将加大选择偏差'),
        dict(item='历史正式投影CNN选择',decision='有条件保留，仍需验证',evidence='本轮维持既有正式选择，仅完成证据审计，不证明其全面优于插值',cost='不改变选择文件',risk='课程任务五明显优于插值标准仍部分满足')]
    write_json(out/'decisions.json',decisions)
    new_cross=all_metrics[(all_metrics.split=='new_test')&(all_metrics.comparison=='cross_repeat')]
    selected_summary=summary[(summary.negative=='allowed_known')&(summary.repeat==1)][['hidden_type','method','n_hidden','n_negative','auc','low','high']]
    gate_summary=fit_table.groupby('hidden_type').agg(folds=('fold','size'),fusion_enabled=('fusion_enabled','sum'),reference_min=('known_references','min'),reference_max=('known_references','max'),validation_reference_min=('validation_references','min'),validation_reference_max=('validation_references','max')).reset_index()
    stratum_counts=improvements[(improvements.split=='new_test')&(improvements.comparison=='cross_repeat')&(improvements.reference=='train_distance_prior')&(improvements.method=='allocation_shared_same')]
    error_keys=['axis','distance_lower_bp','distance_upper_bp','layer_lower','layer_upper','metric','delta','reference_error','descriptive_percent_reduction']
    stratum_details=relative[(relative.split=='new_test')&(relative.comparison=='cross_repeat')&(relative.method=='allocation_shared_same')&relative.metric.isin(['raw_mse','conditional_proportion_mse'])][error_keys]
    stratum_details.to_csv(out/'shared_stratum_effect_details.csv',index=False)
    cnew=composition[(composition.split=='new_test')&(composition.axis=='count')]
    far_share=float(cnew.loc[cnew.distance_lower_bp>=1000,'pixel_fraction_within_axis'].sum())
    sparse_share=float(cnew.loc[cnew.layer_upper<=4,'pixel_fraction_within_axis'].sum())
    rho=corr_summary[(corr_summary.method=='ae_residual')&(corr_summary.hidden_type==corr_summary.actual_type)][['hidden_type','nuisance','finite_folds','median_spearman','max_absolute_spearman','minimum_case_count','maximum_case_count']]
    delta_auc=auc[(auc.repeat==1)&(auc.negative_set=='allowed_known')&(auc.method=='ae_residual')][['hidden_type','positive','negative','delta','low','high']]
    prior_delta=example.set_index('metric')
    occlusion_path=root/'course_audit/committed_evidence/results/classification_fine/explanations/occlusion_summary.csv'
    occlusion=pd.read_csv(occlusion_path,header=[0,1],index_col=[0,1]);center_drop=float(occlusion.loc[('neutral','CHIN'),('center_probability_drop','mean')]);corner_drop=float(occlusion.loc[('neutral','CHIN'),('corner_probability_drop','mean')])
    overlap_count=len(pd.read_csv(root/'novelty_diagnosis/annotation_overlap_pairs.csv'))
    ds_counts=pd.read_csv(root/'sr_audit/known_structure_sample_counts.csv');downstream=pd.read_csv(root/'sr_audit/known_structure_all_methods.csv')
    dsnew=downstream[(downstream.split=='new_test')&(downstream.comparison=='cross_repeat')&downstream.method.isin(['train_distance_prior','allocation_shared_same','bicubic_unprojected','bilinear_unprojected'])]
    check_rows=[]
    for path in sorted((root/'runs').glob('*.json')):
        rec=json.loads(path.read_text(encoding='utf-8'))
        if rec['status']!='running':check_rows.append(dict(stage=path.stem,status=rec['status'],exit_code=rec['exit_code'],seconds=rec.get('elapsed_seconds')))
    pd.DataFrame(check_rows).to_csv(out/'completed_runs_at_report.csv',index=False)
    label_summary=labels.groupby('workbook').agg(rows=('start','size'),same_CHIN_endpoints=('minimum_CHIN_endpoint_L1_bp',lambda x:int((x==0).sum())),near_CHIN_endpoints_50bp=('minimum_CHIN_endpoint_L1_bp',lambda x:int((x<=50).sum())),overlaps_any_known=('overlapping_known',lambda x:int((x>0).sum())),all_within_genome=('within_genome','all')).reset_index()
    body=f'''# 第七轮：课程差距、独立证据与冻结模型审计

## 结论与范围

本轮已实际完成原始要求核对、来源核验、全部候选匿名包、逐类逐折诊断、强基线统一比较、冻结权重分层推理及工具检查。**没有新增模型训练，没有发现合格新结构簇，没有证据支持宣布超分辨获得实质性多指标突破。** 独立人工审核仍待完成：{human['reviewed']}/{human['cases']}例。工程工作完成与科学证据充分必须分别评价。

历史空间划分、评分、阈值、候选预算和正式选择保持不变。本轮结果为回顾性诊断；新空间组也处在此前已扫描的同一基因组，不能称为新的生物学独立测试。未推送 GitHub，未整理或替换正式报告/PPT/视频/提交包。

## A. 教师原始要求与课程评分差距

依据：上级目录教师 PDF，逐页提取与哈希见 [原始材料记录](course_audit/teacher_pages.json)，关键页截图见 `course_audit/teacher_page_*.png`。任务一40、任务二30、任务三10为必做；选做部分20分，任务四（条件分类或差异定位，p11）、任务五（超分辨，p12）、任务六（三维重建及MDS等基线，p12–13）至少选一。仓库实验计划已明确选择任务五，本轮不将未选的四/六计为必做缺口，也不擅自改选方向。总评分维度为问题理解15%、方法设计25%、实现20%、实验评价25%、结果解释15%（p14）；这些维度不能与任务分数机械相乘预测得分。

**任务二可以在没有特别好的新发现时完成最低课程流程。** p10写的是正确识别已知模式与复现典型结构；没有要求必须产生若干新簇。示例簇数、成员数和一致性示例不是必须达到的性能门槛。可以按“已完成发现流程，未证明新结构”结项；不能写成成功发现新结构或全部高分标准已满足。

**任务五还不能充分声称达到“明显优于插值且稳定恢复已知形态”。** 严格区块守恒是本项目的工程选择，教师没有要求所有方法都必须投影。不能只比较投影后较弱插值以回避原始插值的形态优势。

{table(gaps)}

### 本轮优先投入与必做任务旁证

原始排序及成本/收益见 [priorities.json](course_audit/priorities.json)：全候选复核、强插值审计、独立来源核验、类别/计数失败诊断，以及任务一/三和提交缺口核对。

任务一：重新汇总{classification['n']}条、{classification['groups']}空间组的既有折外预测，accuracy={classification['accuracy']:.6f}，macro-F1={classification['macro_f1']:.6f}；CHID precision={classification['report']['CHID']['precision']:.6f}、recall={classification['report']['CHID']['recall']:.6f}，样本{classification['report']['CHID']['support']:g}。整体准确率不能证明少数类别稳定。既有遮挡摘要中，中性填充下CHIN中心遮挡概率下降{center_drop:.6f}、角落{corner_drop:.6f}；背景影响未排除，不能仅凭显著图就断言只关注结构。零填充结果也对分布变化敏感。

任务三：两个已提交轨道集各465张，索引均连续覆盖0至4,641,652 bp；协议具备至少三条件、条件曲线/均值/基因/已知结构四轨。当前稀疏检出没有全图，本轮按固定提交获取每套首/中/末共六张并核对Git对象；抽样可读，不冒称930张全部视觉审阅。局部接触是±10 kb窗口信号，不是全染色体总覆盖。见 [索引审计](course_audit/track_inventory.json)、[抽样来源](material_verification/track_samples.json)。

## B. 任务二：独立来源、全候选与新颖性失败原因

### B1. 独立数据和标签

本地清单29项，11份完整条件数据包含既用WT两重复及44C、FIS、HNS、rif25、NoLig等；它们已参与历史探索或候选描述，不能升级为未见独立验证。同一基因组换窗口也不等于新重复。所有候选的11条件审核沿用第六轮，不重复下载与计算。

GEO三个系列的{geo['metadata_rows']}条元数据记录去重为{geo['unique_geo_accessions']}个GSM，而不是99个独立重复。WT四个t1/t2文库记录同为MG1655、NC_000913.3，个体附属矩阵字段为NONE，系列提供的仍是rep1/rep2矩阵。尚不能确认四个文库与已用两份矩阵的合并关系；不将标题当作新增独立数据。Hi-C是不同实验模态，RNA/Red-C可作功能背景，均不能直接作为深覆盖10 bp Micro-C真值。没有下载大型数据。

作者最新提交的三类BED与原固定版本逐字节一致，仍为26 CHID、250 CHIN、68 OPCID。同作者两个小型工作簿已经下载核对；坐标兼容性如下，额外工作簿的具体实验条件、标注流程及独立性仍未从工作簿本身确认。早期机器生成元数据中的“condition-specific”只是保守使用限制，不是已证实的实验条件；不把文件名“newly emerged”解释为独立WT新结构真值。

{table(label_summary)}

即使坐标在同一参考序列范围内、与旧注释不重叠，也不证明适用WT或属于新类。最值得补充的是：文库到处理后矩阵的确切来源映射、未参与选择的匹配WT深测序矩阵，以及有人员/条件/定义来源的嵌套或边界标签。

来源与实际覆盖/限制见 [本地数据表](independent_data/local_data_inventory.csv)、[逐条远程记录](independent_data/remote_sample_inventory.csv)、[兼容性结论](independent_data/compatibility_summary.json)、[补充表坐标核验](review/supplemental_coordinate_compatibility.csv)。原始元数据来自 [GSE272159](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272159/soft/GSE272159_family.soft.gz)、[GSE272158](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272158/soft/GSE272158_family.soft.gz)、[GSE272161](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272161/soft/GSE272161_family.soft.gz)，作者标签见 [ecoli_microc](https://github.com/irzhegalova/ecoli_microc)；每次保存的具体URL、提交和哈希均在本轮来源表中。

### B2. 全候选冻结复核

从第六轮冻结评分数组按原NMS与预算重新生成{blind['regenerated_records']}条候选记录，逐行核对得分、排名和坐标完全一致；去重为{blind['unique_cases']}个中心，而不是243个新结构。每例包含WT两重复、原生支持派生的有效像素、训练背景中按密度/零值/支持率匹配的对照、相对位置注释上下文和实际100 bp粗粒度矩阵。矩阵是每百万归一化图像信号，不是假称原生10 bp独立读段；原生源文件见来源清单。显示比例和色标统一。

公开包不含算法、得分、排名和绝对坐标；坐标/算法映射与匿名密钥保存在private目录。注释上下文是已知元数据，既非审核结论也非新结构标签。匿名不能保证熟悉该基因组的人无法凭形态识别区域。

- 可交给独立人员的文件：[blind_review_packet.zip](blind_review/blind_review_packet.zip)，解压后打开index.html；也可直接打开 [本地复核页](blind_review/public/index.html)。
- 不要把整个blind_review目录交给复核者；private及完整研究报告包含解盲信息。
- 每例须填写人员、明确类别和依据；记录两个重复、边界、背景及不确定性。七种复核类别区分已知形态、可能边界变化、可能嵌套、邻近、背景、独特可重复候选和证据不足。
- 网页导出JSON，或填写review_template.csv；使用 `scripts/review_blind_candidates.py --import-json 文件` / `--import-csv 文件` 导入真实记录。追加日志记录实际导入UTC、原始客户端时间、材料哈希及前一条哈希；同例多人分歧保留，不反向改分数。

独立性须有真实人员声明；**当前人工审核0例，独立验证未完成。** 实际Edge无网络页面测试、243份图片/矩阵哈希与可读性检查、ZIP完整性通过。临时浏览器工程记录已丢弃，没有进入人工日志。见 [人工状态](blind_review/human_review_status.json)、[工具检查](material_verification/status.json)。

### B3. 嵌套与邻近：保留分类问题，不放松新簇规则

原{len(known)}条注释间有{overlap_count}对区间重叠，包括CHID/CHIN与CHIN内部包含关系，说明几何重叠确实存在。不能从相交关系推导不同生物学机制或新类。{blind['unique_cases']}候选的25.6 kb上下文几何分布为：

{table(geometry)}

上下文含已知结构的窗口可能仍有值得研究的边界或嵌套形态；本轮给它们单独待审入口。旧的完整上下文+200 bp排除、覆盖、一致性、独立空间组及聚类规则不变，旧标准下仍0个合格新簇。private表中不含注释的布尔字段只描述直接相交，不替代含200 bp缓冲的正式排除；几何表外部区域也不自动合格。见 [原注释重叠表](novelty_diagnosis/annotation_overlap_pairs.csv)。

### B4. 按类别与空间折的误差诊断

以下为第六轮真实结果复用：隐藏类vs允许已知类，WT1，按空间组区间。全部重复、背景负例及逐折结果另附，不能以隐藏vs普通背景的较高AUROC代替新颖性辨别。

{table(selected_summary)}

逐折范围：

{table(fold_range)}

训练/参考数量和验证门控：

{table(gate_summary)}

根本限制是：背景重建误差测“难重建/异常”，不等于“区别于所有已知形态”；AE残差在CHIN尤其弱，而加入已知参考距离的诊断对该类有信号，但没有全类别稳定优势。某些CHID折只有极少正例，不能据单折高AUROC下结论；留出CHIN时允许参考更少，门控与校准也更不稳定。联合方案只在少数OPCID折通过验证，其余回退AE；不能把回退后的结果当成普遍互补性的证明。

本轮新增了同例空间组配对AUROC差值及区间，见 [paired_auc_differences.csv](review/paired_auc_differences.csv)；没有据这些回顾性结果调整权重或参考规则。去掉方法重复后的类别密度、零值、支持与有效像素分布见 [unique_class_nuisance.csv](review/unique_class_nuisance.csv)；各类别各折的相关性见 [完整相关表](novelty_diagnosis/within_class_nuisance_correlations.csv)。即使做过训练背景的强度校准，残差与密度/稀疏度仍可能关联；相关性不是因果归因，也不是已证明只识别形态。样本过少或支持率恒定时保留未定义值。原AE原始误差及校准后分布均保留；fits中继承的pca_components/paired_training_mse字段是旧表示诊断，不冒充AE训练指标，真实AE训练看seed_diagnostics、权重和曲线。

AE残差相对形态基线的配对AUROC差值（WT1，隐藏类vs允许已知类）：

{table(delta_auc)}

隐藏类别内部的AE残差与干扰量相关性（各折、两重复的描述性汇总；小样本的极端相关不作强证据）：

{table(rho)}

**与第六轮的对照：算法、训练和正式候选数均未改善；证据流程与失败定位得到补强。** 不把工具改进写成新颖性性能提高。当前最强结论是“可复现筛选与已知形态复现流程已具备；现有证据不支持新结构发现”。

## C. 任务五：强基线、多指标取舍与可恢复信息

### C1. 统一比较及不确定性

整窗全部{sr_status['main_rows']}条结果复用第六轮，包括两个种子的逐例指标，种子先取均值，不当作独立空间组，也不构造未经验证的网络集成。下表是新空间组跨重复双向平均；同重复、历史组、分方向与原始误差/计数偏差全部见 [统一表](sr_audit/unified_all_metrics.csv)、[双向表](sr_audit/directional_metrics.csv)、[全部配对区间](sr_audit/all_paired_intervals.csv)。数值差为方法减参照，PSNR/SSIM越大越好，误差越小越好；只有6个新空间组，区间也是回顾性且未作多重比较校正。

{table(new_cross[['method','psnr','ssim','raw_mae','raw_nrmse','oe_50bp_ssim','oe_100bp_ssim','oe_160bp_ssim']])}

train_distance_prior相对未投影双三次的全部主要指标差值：

{table(example[['metric','delta','low','high','groups']])}

PSNR差{prior_delta.loc['psnr','delta']:+.6f} dB，原生SSIM差{prior_delta.loc['ssim','delta']:+.6f}且区间跨零；100 bp和160 bp形态差分别为{prior_delta.loc['oe_100bp_ssim','delta']:+.6f}、{prior_delta.loc['oe_160bp_ssim','delta']:+.6f}。MAE与50 bp可改善，但稳定化Poisson偏差明显变差：严格守恒输出在源为零、目标有计数时被迫为零，数值地板会产生大偏差。因此不能只按PSNR或MAE宣传更好。未投影双线性在部分跨重复形态尺度更强，已保留；它的源聚合偏差也是实际取舍，不能据此将它从课程比较中删除。

![全部强基线配对区间](review/strong_baseline_intervals.png)

正式选择文件仍是 `{frozen['formal_choices']['superresolution']['primary']}`，不是train_distance_prior。本轮不把实验验证选择直接升级为正式默认，也不声称历史正式选择已被证明全面最佳。

### C2. 守恒与尺度冲突的数学原因

输入低分辨率定义为4×4原生图像像素的**均值**，不是唯一读段的总计数。严格守恒要求每块输出均值等于输入（图像块总量为16倍输入）。若源块为0且输出非负，则该块16个输出全为0；不能同时恢复另一重复在该处出现的正计数。此限制是目标与约束的冲突，不能单靠增加网络容量解决。

160 bp是40 bp的整数倍且本协议对齐，160 bp聚合由输入块总量完全确定。各守恒方法160 bp O/E SSIM最大差仅{sr_status['aligned160bp_constrained_max_method_range']:.3g}，属于数值误差；这个尺度上的相同或改善不能算网络学到了新结构。50/100 bp未完全对齐，可以受块内分配影响。守恒/非负/对称审计见 [constraints_by_method.csv](sr_audit/constraints_by_method.csv)：守恒方法相对误差在约1e-6内；原始插值保留为非守恒对照。零源分母地板会放大原始插值相对误差，不应将巨大相对数值解释为同倍数的绝对物理偏差。

### C3. 所有计数、距离和有效像素层

已用冻结权重重新推理主要十二种方法，两种子全部保留，共{sr_status['stratified_rows']}条分层记录；{sr_status['predictions_verified_against_frozen_rows']}次整窗原始MAE与历史冻结值逐一匹配。不是新训练或新方法筛选。每个距离区间均有六个计数层、四个有效像素层，所有空层和未定义的条件误差保留。原始MAE/MSE按唯一上三角原生像素加权，含零目标；条件比例误差仅比较源与目标都为正的区块，不能代表所有稀疏区域。有效像素层是两个重复的观测正像素比例，不是独立覆盖/可映射性掩码；层变量观察了两重复，只作回顾性诊断。

共享网络相对训练距离先验的分层误差统计（新空间跨重复；误差差值越小越好）：

{table(stratum_counts[['axis','metric','defined_cells','empty_or_undefined','point_estimate_lower','interval_entirely_lower','interval_entirely_higher']])}

这些格子不是独立假设检验，不能按“显著格子数”宣布成功。完整效应、绝对误差、描述性相对降幅及各层像素占比见 [全部分层配对区间](sr_audit/stratified_paired_intervals.csv)、[误差与相对量](review/relative_stratum_effects.csv)、[层构成](review/stratum_composition.csv)。分层误差不冒充局部SSIM。

按新空间组实际唯一像素构成，距离≥1000 bp占{far_share:.2%}，平均图像块计数<4占{sparse_share:.2%}。共享网络相对先验的较大相对降幅主要在近距离、低到中等计数/有效像素层，远距离大多很小或退化；并不是“只在最高计数才有效”。完整逐层绝对量与相对量见 [shared_stratum_effect_details.csv](review/shared_stratum_effect_details.csv)。这只是冻结网络相对先验的小幅局部分配改进，不能推导多数稀疏像素恢复了可重复细节。

![全分层收益与空层](review/all_strata_effects.png)

复用{noise.sample_index.nunique()}个训练/验证窗口的原噪声诊断；没有再统计同一份大型区块缓存。所有{len(noise_summary)}个计数×距离单元，包括空层，见 [复用噪声汇总](review/reused_noise_summary_all_cells.csv)。稀疏/远距离区域往往缺少共同正区块，低分辨率本身与细节都更不稳定；高计数不保证所有距离的细节一致。请结合该表的实际相关性、共同正区块数和源零比例，而不是把某一组高相关外推至全图。按两个重复的平均计数切层会引入范围限制及选择关联，层内负相关不能直接解释成真实接触反相关。两个重复不足以分离生物差异和采样噪声，经验相关性不是理论信息上限。

均匀分配、训练距离先验及网络的同层对照都已给出；若先验已经接近网络，收益首先应归因于分配先验，而不能泛称学习到可泛化细节。部分分层有小收益、部分退化，整窗跨重复多尺度没有一致优势。当前证据不支持仅在高计数挑样，也不支持承诺稳定恢复真实10 bp细节。

### C4. 下游边界

局部完整注释只有：

{table(ds_counts)}

新空间组跨重复，四个关键方法的局部形态/峰位置指标：

{table(dsnew[['method','type','oe_100bp_ssim','oe_100bp_pearson','signal_profile_peak_displacement_bp']])}

所有同/跨重复及历史组、全部模型保存在 [下游汇总](sr_audit/known_structure_all_methods.csv)。这是10.24 kb原生片段上的19条局部注释，CHID仅1条；原分类器要求12.8/25.6 kb上下文，不能将形态相似度、100 bp峰位移或旧的49条完整上下文结果冒充新模型分类准确率。当前无法据此证明“稳定恢复各类结构识别能力”。

### C5. 停止决策

共享训练vs匹配单重复的100 bp收益极小，同时PSNR略降（逐例旧结果与第六轮配对表保留）；相对先验也非多指标一致收益。现有证据指向目标噪声、守恒/跨重复冲突和可辨识性，不指向容量不足。本轮没有启动新的架构/损失/权重搜索。若未来要调整软守恒或多尺度目标，需先说明新的输入物理语义、训练/验证选择及独立验证数据，不能用本轮历史测试结果调权。

## D. 工程、复现与最终保护

- 本轮新增配置 `configs/evidence_review_v1.json`，继承固定第五/六轮协议；[PROTOCOL.md](PROTOCOL.md)说明冻结与回顾性边界。
- 新模块 `src/dlpro/evidence_review.py` 负责匿名ID、几何关系、追加式记录和唯一像素分层统计；四项针对性测试覆盖身份稳定/私钥、几何非生物标签、记录完整性与篡改、零目标误差与对角唯一像素。
- 阶段入口 `scripts/run_evidence_review.py`；审计、元数据、候选、诊断、超分辨、材料和汇总均保存命令、真实退出码、源码快照、配置、环境及日志。此前失败/中断记录保留在历史轮次；本轮全部尝试见最终运行清单，失败不覆盖。
- 训练权重没有更新；本轮推理使用的权重哈希在 [evaluation_lock.json](sr_audit/evaluation_lock.json)，候选选择哈希在 [selection_freeze.json](blind_review/selection_freeze.json)。源大数据哈希复用已验证的历史来源记录，本轮没有冒称重新下载或全部重哈希。
- 测试日志见 `runs/test*.log`，浏览器与真实矩阵检查见 `material_verification/`。报告生成后的最终校验见 `verification/final_status.json`、`verification/run_status.csv`、`verification/change_manifest.csv`：验证原保护清单、空间隔离、权重/源码快照、匿名包和正式选择；它是最终状态依据。
- 复现时从仓库根目录使用本地Python，设置PYTHONPATH=src；先audit、metadata、blind、novelty，再superresolution、materials、test、report；本轮目录拒绝覆盖已完成阶段。要重跑先选择新的输出目录，不覆盖正式选择。模型及原数据路径沿用第五/六轮来源表，未经完整数据准备不能把轻量检出当作正式运行。

## E. 保留、停止和下一步

{table(pd.DataFrame(decisions))}

最影响得分的三个科学问题及后续投入顺序：

1. **任务一必做40分：少数CHID识别与解释性证据不足。** 已定位弱类和背景敏感旁证；本轮未重训，仍需有冻结空间验证的定向研究。优先于继续投入低成功率的超分辨架构搜索。
2. **任务二必做30分：新颖性证据、形态/强度区分与独立审核不足。** 已解决完整复核材料、来源误计和分类/空间折诊断；真实独立人员和新匹配数据仍缺。建议先用全候选包收集带依据与分歧的真实审核，再核实外部WT来源；不降低新簇阈值。没有新发现仍可按已知结构复现与流程标准结项，但不能按发现新类表述。
3. **任务五选做20分：缺少相对合理强插值的实质多指标优势。** 已补齐强基线、配对区间、全部分层、约束数学边界和下游样本限制；数据限制尚未解决，停止继续网格调结构/损失。保留诚实的收益/退化对照更有研究价值。

最终提交的代码ZIP、正式报告、PPT/视频、组员贡献与命名仍是另一个交付门槛；本轮没有擅自修改或认定这些正式成果已完成。当前最值得推进的是实际独立复核和必做任务证据整理，而不是继续积累没有独立验证的实验数量。
'''
    (root/'REPORT.md').write_text(body,encoding='utf-8',newline='\n')
    write_json(out/'status.json',dict(report_generated=True,new_models_trained=0,formal_choices_changed=False,confirmed_new_clusters=0,independent_human_review_completed=False,paired_auc_rows=len(auc),all_fold_rows=len(folds),no_claim_of_material_sr_breakthrough=True))
    print('Evidence report and explicit keep/stop decisions generated from frozen artifacts',flush=True)

if __name__=='__main__':main()
