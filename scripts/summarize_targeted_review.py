"""Generate comparisons, decisions and integrity checks from actual stage artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.method_review import load_config,sha256,write_json
from dlpro.novelty import grouped_auc
from dlpro.targeted_review import supervision_repeats
from evaluate_allocation_review import interval
from summarize_method_review import paired_auc_difference

def md(frame):
    columns=list(map(str,frame.columns))
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'review';out.mkdir(exist_ok=True)
    dc=cfg['discovery'];ae=root/dc['result_subdir'];ev=root/'evaluation';previous=Path(cfg['previous_output'])
    points=pd.read_csv(ae/'point_scores.csv');recall=pd.read_csv(ae/'budget_recall.csv');novel=pd.read_csv(ae/'novelty_summary.csv');fits=json.loads((ae/'fits.json').read_text(encoding='utf-8'))
    incumbent=pd.read_csv(previous/'novelty_corrected/point_scores.csv');keys=['hidden_type','fold','repeat','method','item_id']
    comparison=points[points.method=='morphology_baseline'].merge(incumbent[incumbent.method=='morphology_baseline'],on=keys,validate='one_to_one',suffixes=('_new','_old'))
    assert len(comparison)==len(incumbent[incumbent.method=='morphology_baseline'])
    score_difference=float(abs(comparison.score_new-comparison.score_old).max());assert score_difference<1e-7
    recalls=[]
    for key,part in recall.groupby(['hidden_type','repeat','method','global_equivalent_budget']):
        for kind,sub in [('ALL',part),*list(part.groupby('actual_type'))]:
            delta=sub.recalled.astype(float).to_numpy()-sub.random_hit_probability.to_numpy()
            estimates=interval(delta,sub.group,cfg)
            recalls.append(dict(zip(['hidden_type','repeat','method','global_equivalent_budget'],key),actual_type=kind,center_recall=float(sub.recalled.mean()),random_recall=float(sub.random_hit_probability.mean()),n=len(sub),**estimates))
    recall_summary=pd.DataFrame(recalls);recall_summary.to_csv(out/'recall_vs_random.csv',index=False)
    auc_differences=[]
    for (hidden,repeat,method),part in points.groupby(['hidden_type','repeat','method']):
        if method=='morphology_baseline':continue
        base=points[(points.hidden_type==hidden)&(points.repeat==repeat)&(points.method=='morphology_baseline')]
        paired=part.merge(base,on=['item_id','group','actual_type'],validate='one_to_one',suffixes=('_method','_base'))
        for negative in ('allowed_known','background'):
            sub=paired[(paired.actual_type==hidden)|((paired.actual_type=='background') if negative=='background' else ((paired.actual_type!=hidden)&(paired.actual_type!='background')))]
            estimate=paired_auc_difference(sub.actual_type.eq(hidden).to_numpy(),sub.score_method.to_numpy(),sub.score_base.to_numpy(),sub.group.to_numpy(),cfg)
            auc_differences.append(dict(hidden_type=hidden,repeat=repeat,method=method,negative=negative,**estimate))
    pd.DataFrame(auc_differences).to_csv(out/'novelty_vs_morphology.csv',index=False)
    selected=pd.read_csv(ae/'selected_scan_windows.csv');repeat_rows=[]
    for key,part in selected.groupby(['hidden_type','method','global_equivalent_budget']):
        one=set(part[part.repeat==1].center);two=set(part[part.repeat==2].center)
        repeat_rows.append(dict(zip(['hidden_type','method','global_equivalent_budget'],key),rep1=len(one),rep2=len(two),jaccard=len(one&two)/len(one|two)))
    pd.DataFrame(repeat_rows).to_csv(out/'candidate_repeat_stability.csv',index=False)
    sr=pd.read_csv(ev/'summary.csv');effects=pd.read_csv(ev/'paired_intervals.csv');checks=json.loads((ev/'checks.json').read_text(encoding='utf-8'));selection=json.loads((ev/'validation_selection.json').read_text(encoding='utf-8'))
    downstream=pd.read_csv(ev/'known_structure_per_window.csv')
    downstream_effects=[]
    if len(downstream):
        dm=downstream.groupby(['split','comparison','method','group','start','source_repeat','structure_id','type'])[['oe_100bp_ssim','oe_100bp_pearson','signal_profile_peak_displacement_bp']].mean().reset_index()
        for (split,comparison),part in dm.groupby(['split','comparison']):
            left=part[part.method=='allocation_shared_same']
            for reference in ['allocation_single_same','train_distance_prior','projected_bicubic','allocation_expanded_mixed']:
                pair=left.merge(part[part.method==reference],on=['group','start','source_repeat','structure_id','type'],suffixes=('_a','_b'),validate='one_to_one')
                for metric in ['oe_100bp_ssim','oe_100bp_pearson','signal_profile_peak_displacement_bp']:
                    use=pair[[metric+'_a',metric+'_b']].notna().all(1);valid=pair[use]
                    if len(valid):downstream_effects.append(dict(split=split,comparison=comparison,reference=reference,metric=metric,annotations=int(valid.structure_id.nunique()),rows=len(valid),**interval((valid[metric+'_a']-valid[metric+'_b']).to_numpy(),valid.group,cfg)))
    pd.DataFrame(downstream_effects).to_csv(out/'known_structure_paired_intervals.csv',index=False)
    # Reconstruct the exact documented RNG streams to report actual pair exposure.
    plan=pd.read_csv(root/'data/sr_windows.csv');pool=np.flatnonzero(plan.split=='train');exposures=[]
    for seed in cfg['sr']['train_seeds']:
        for arm in cfg['sr']['arms']:
            spatial=np.random.default_rng(seed);repeat_rng=np.random.default_rng(seed+cfg['seed']);counts=np.zeros((len(plan),2),int)
            for _ in range(cfg['sr']['steps']):
                ids=spatial.choice(pool,cfg['sr']['batch_size'],replace=False);spatial.random(len(ids))
                repeats=supervision_repeats(arm,repeat_rng,len(ids));np.add.at(counts,(ids,repeats),1)
            for idx in pool:exposures.append(dict(method=arm,seed=seed,sample_index=int(idx),start=int(plan.iloc[idx].start),group=int(plan.iloc[idx].group),repeat1_draws=int(counts[idx,0]),repeat2_draws=int(counts[idx,1])))
    pd.DataFrame(exposures).to_csv(out/'actual_training_pair_exposure.csv',index=False)
    quality=pd.read_csv(ae/'candidate_quality_summary.csv');clusters=pd.read_csv(ae/'cluster_quality.csv')
    conditions=json.loads((root/'conditions/protocol.json').read_text(encoding='utf-8'));noise=pd.read_csv(root/'noise/stratified_repeat_agreement.csv')
    noise_summary=noise.groupby(['distance_lower_bp','distance_upper_bp','count_lower','count_upper']).agg(total_blocks=('blocks','sum'),both_positive_blocks=('both_positive_blocks','sum'),mean_window_input40_pearson=('input40_pearson','mean'),defined_input40_correlations=('input40_pearson','count'),mean_window_common_positive_detail_pearson=('normalized_detail_pearson_common_positive','mean'),defined_detail_correlations=('normalized_detail_pearson_common_positive','count'),mean_window_proportion_mse=('proportion_mse_all_blocks','mean')).reset_index()
    noise_summary.to_csv(out/'noise_strata_summary.csv',index=False)
    nuisance=pd.read_csv(ae/'nuisance_correlations.csv');seed_rows=pd.read_csv(ae/'reconstruction_per_window.csv')
    seed_rows.groupby(['hidden_type','seed','repeat','actual_type'])[['raw_mse','calibrated_residual','residual_rank']].agg(['mean','median','std','count']).to_csv(out/'reconstruction_by_category.csv')
    nuisance.to_csv(out/'nuisance_correlations.csv',index=False)
    seed_summary=[]
    for key,part in seed_rows[seed_rows.actual_type!='background'].groupby(['hidden_type','seed','repeat']):
        hidden,seed,repeat=key;y=part.actual_type.eq(hidden).to_numpy()
        from sklearn.metrics import roc_auc_score
        seed_summary.append(dict(hidden_type=hidden,seed=seed,repeat=repeat,auc_hidden_vs_allowed=float(roc_auc_score(y,part.residual_rank))))
    pd.DataFrame(seed_summary).to_csv(out/'ae_seed_stability.csv',index=False)
    # Compare shared exposure to the matched single-repeat arm, not just to interpolation.
    primary=effects[(effects.method=='allocation_shared_same')&(effects.reference=='allocation_single_same')]
    primary.to_csv(out/'shared_vs_single.csv',index=False)
    newcross=primary[(primary.split=='new_test')&(primary.comparison=='cross_repeat')&(primary.metric=='oe_100bp_ssim')].iloc[0]
    historicalcross=primary[(primary.split=='historical_test')&(primary.comparison=='cross_repeat')&(primary.metric=='oe_100bp_ssim')].iloc[0]
    same50=primary[(primary.split=='new_test')&(primary.comparison=='same_repeat')&(primary.metric=='oe_50bp_ssim')].iloc[0]
    shared_morphology_benefit=bool(newcross.low>0 and historicalcross.low>0)
    # A favorable matched mean is insufficient to replace the currently retained prior.
    versus_prior=effects[(effects.method=='allocation_shared_same')&(effects.reference=='train_distance_prior')]
    priorcross=versus_prior[(versus_prior.split=='new_test')&(versus_prior.comparison=='cross_repeat')&(versus_prior.metric=='oe_100bp_ssim')].iloc[0]
    novelty_primary=novel[(novel.method=='ae_residual')&(novel.negative=='allowed_known')&(novel.repeat==1)]
    diff=pd.DataFrame(auc_differences);novelty_effect=diff[(diff.method=='ae_residual')&(diff.negative=='allowed_known')&(diff.repeat==1)]
    # Require improvement for all held types, rather than picking the best category.
    novelty_robust=bool(len(novelty_effect)==len(novelty_primary) and novelty_effect['low'].gt(0).all() and novelty_primary['low'].gt(.5).all())
    protections=json.loads((root/'protected_existing_files.json').read_text(encoding='utf-8'));protected=all(Path(name).exists() and sha256(name)==value for name,value in protections.items())
    assert protected
    run_rows=[];archives=True
    for path in sorted((root/'runs').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        if 'command' not in record:continue
        source=Path(record['source_archive']);archives &= sha256(source)==record['source_archive_sha256']
        with zipfile.ZipFile(source) as z:
            for name,value in record['code_sha256'].items():archives &= hashlib.sha256(z.read(name)).hexdigest()==value
        run_rows.append(dict(stage=path.stem,status=record['status'],exit_code=record['exit_code'],elapsed_seconds=record.get('elapsed_seconds'),command=' '.join(record['command'])))
    assert archives
    write_json(out/'method_provenance.json',dict(fixed_protocol_sha256=sha256('docs/第六轮定向优化固定协议.md'),inherited_partition_fits_sha256=sha256(previous/'novelty_corrected/fits.json'),fit_metadata_note='fits.json preserves inherited fifth-round partition/reference diagnostics; current AE training is solely background_training_indices and seed_diagnostics/selection_lock, not inherited PCA or paired_training_mse fields',old_historical_native_rows_reused=True))
    pd.DataFrame(run_rows).to_csv(out/'run_status.csv',index=False)
    # Plots show effects and full category failures, rather than only favorable values.
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for ax,(hidden,part) in zip(axes,novel[(novel.repeat==1)&(novel.negative=='allowed_known')].groupby('hidden_type')):
        ax.errorbar(part.auc,np.arange(len(part)),xerr=np.stack([part.auc-part.low,part.high-part.auc]),fmt='o');ax.set_yticks(np.arange(len(part)),part.method,fontsize=8);ax.axvline(.5,color='grey',ls='--');ax.set_title(hidden);ax.set_xlim(0,1);ax.set_xlabel('Hidden vs allowed-known AUROC / spatial95% interval')
    fig.savefig(out/'novelty_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),constrained_layout=True)
    for ax,metric in zip(axes,['psnr','ssim','oe_100bp_ssim']):
        part=primary[(primary.split=='new_test')&(primary.metric==metric)]
        ax.errorbar(part.delta,np.arange(len(part)),xerr=np.stack([part.delta-part.low,part.high-part.delta]),fmt='o');ax.set_yticks(np.arange(len(part)),part.comparison);ax.axvline(0,color='grey',ls='--');ax.set_title(metric);ax.set_xlabel('Shared minus matched repeat1-only')
    fig.savefig(out/'shared_training_effects.png',dpi=150);plt.close(fig)
    params=np.load(previous/'sr_corrected/normalization.npz');expect=params['expected_100']
    for path in sorted(ev.glob('probe_*.npz')):
        data=np.load(path);names=['source','target','bicubic','prior','shared'];fig,axes=plt.subplots(1,5,figsize=(13,3),constrained_layout=True)
        n=data['source'].shape[-1]//10;d=abs(np.arange(n)[:,None]-np.arange(n)[None])
        for ax,name in zip(axes,names):
            field=data[name][:n*10,:n*10].reshape(n,10,n,10).mean((1,3));field=np.log1p(field/expect[d]);field[d<2]=0
            im=ax.imshow(field,origin='lower',cmap='magma',vmin=0,vmax=2);ax.set_title(name);ax.set_xlabel('100 bp bin')
        fig.suptitle(path.stem+' / fixed random spatial probe');fig.colorbar(im,ax=axes,label='log(1+O/E), train-only expectation',shrink=.75);fig.savefig(out/(path.stem+'.png'),dpi=120);plt.close(fig)
    cluster_table=clusters.assign(non_noise=clusters.cluster.ge(0)).groupby('algorithm').agg(group_instances=('non_noise','sum'),provisional_clusters=('provisional_shape_cluster','sum')).reset_index()
    validation_table=pd.DataFrame([dict(hidden_type=f['hidden_type'],fold=f['fold'],background=len(f['background_training_indices']),reference=len(f['reference_indices']),ae_validation_auc=f['validation_auc']['ae'],morphology_validation_auc=f['validation_auc']['morphology'],joint_validation_auc=f['validation_auc']['joint'],fusion_allowed=f['joint_allowed']) for f in fits]);validation_table.to_csv(out/'validation_choices.csv',index=False)
    decisions=dict(novelty_residual_robust_improvement=novelty_robust,novelty_retained='support-aware morphology' if not novelty_robust else 'AE requires further novelty audit',shared_cross_improved_vs_matched_single=shared_morphology_benefit,validation_selected=selection['selected_method'],shared_new_cross100_vs_prior=priorcross.to_dict(),default_operational_choices_changed=False,new_structure_cluster_instances=int(clusters.provisional_shape_cluster.sum()),external_validation_available=False)
    write_json(out/'decisions.json',decisions)
    audit=dict(protected_existing_artifacts_unchanged=protected,source_archives_match=archives,incumbent_morphology_scores_reproduced=True,incumbent_max_abs_difference=score_difference,known_once_per_hidden_method_repeat=bool(points[points.actual_type!='background'].groupby(['hidden_type','repeat','method']).item_id.nunique().eq(json.loads((ae/'status.json').read_text(encoding='utf-8'))['all_known_available']).all()),new_constraints_pass=checks['new_max_count_error']<1e-5 and checks['new_max_symmetry_error']<1e-5,full_candidate_condition_coverage=conditions['unique_selected_centers']==selected.center.nunique() and conditions['wt_heatmaps']==selected.center.nunique(),all_train_val_windows_in_noise_diagnostics=json.loads((root/'noise/protocol.json').read_text(encoding='utf-8'))['windows']==int(pd.read_csv(root/'data/sr_windows.csv').split.isin(['train','val']).sum()),all_evaluations_exploratory=True)
    assert all(v for k,v in audit.items() if isinstance(v,bool));write_json(out/'integrity.json',audit)
    lines=['# 第六轮定向性能优化：实际结果','',
      '全部位置来自反复观察的同一E. coli基因组。本轮完成实现、实际运行和验证，不把历史空间分组称独立最终测试。原始模型、失败实验、正式选择、报告/PPT/视频/提交包未改，未推送远程。','',
      '## 任务二：背景重建是否产生有用新颖性','',
      '历史AE已真实训练，不是空类定义。本轮五折权重推理及64/128预算排序复现；首次严格容差被极小MAD放大的float32差异触发，保存失败日志后增加排序一致性核验。历史推理数值误差在historical_ae_reproduction.csv，不冒称从头重训复现。','',
      '新方法沿用同一网络，新增支持像素上三角损失、训练背景距离/幅度归一化、训练背景多变量残差校准及严格留类评价。两种子共30模型；重复2只用于评估。背景并不保证无未知结构；隐藏CHIN后独立参考仍很少。残差异常与未知形态并非同一个目标。','',
      '### 验证与冻结选择','',md(validation_table),'',
      '联合权重固定各半，只有允许参考对背景验证AUROC同时超过AE/形态0.02且信号相关<0.9才启用；失败折回退AE。未按隐藏类别标签选模型。潜变量距离仅是单列诊断，不生成额外候选。','',
      '### 全部留类指标（两个方向，含负面结果）','',md(novel),'',
      '背景/已知类型的原始重建误差、校准残差、种子分布见reconstruction_by_category.csv，密度/零值/支持混杂相关见nuisance_correlations.csv。fits.json沿用了第五轮分区元数据；其中旧PCA/paired_training_mse字段不是本轮AE训练结果，当前训练以seed_diagnostics、逐轮曲线和权重为准。','',
      '空间配对AUROC差、分类别召回及随机对照详见novelty_vs_morphology.csv、recall_vs_random.csv。64/128是原726点等效预算，实际可评价540点，折内预算按比例分配；不能与旧全基因组预算召回直接混比。','',
      md(recall_summary[(recall_summary.actual_type=='ALL')&(recall_summary.global_equivalent_budget==64)].drop(columns=['actual_type'])),'',
      '### 全候选审核','',md(cluster_table),'',
      md(quality),'',
      f"覆盖{conditions['unique_selected_centers']}个唯一候选中心，全部生成统一色标WT热图。簇数为留类/重复/方法/预算的累计实例，不是独立新发现。原最少5个独立成员、已知上下文排除、共同有效像素及0.5重复相关标准未降低。",'',
      md(pd.read_csv(root/'conditions/condition_summary.csv',header=[0,1],index_col=0).reset_index().rename(columns=lambda c: str(c))),'',
      '条件形态和强度对完整列表统一计算，未按有利条件挑候选；NoLig单样本只作描述。完整记录见conditions/，无法恢复的完整原件见conditions/protocol.json。方向联合验证不通过的折没有被静默用作推荐。','',
      f"留类残差稳健改善门槛：{novelty_robust}。当前科学证据不能由已知召回、少量高相关候选或单个好类别替代。保留决策见decisions.json。",'',
      '## 任务五：双重复自身监督是否有额外贡献','',
      '第五轮已经共享使用两个重复，本轮只新增唯一上三角同重复NLL的重复1单源与双重复共享匹配对照；相同初值、空间批次、反向增广、231训练窗/21块、320步、两种子和模型。原形态共享、跨重复NLL、混合目标和简单基线全部复用既有权重与同口径逐例输出，不重跑相同训练。','',
      '空目标块无条件分配信息。同重复输入下源零意味着整个原目标块为零；概率仍定义，最终计数严格等于16×块均值。旧conditional_allocation_nll的源正/目标零屏蔽会选中两重复共阳性块，而新唯一接触NLL将源零正目标的潜在概率也纳入跨重复目标；本轮同重复对照避免该源/目标冲突，不声称消除测序噪声。','',
      f"统一验证选择：{selection['selected_method']}；选择与全部模型哈希在新测试推理前锁定。表中种子指标平均，不是选优种子或预测集成。",'',md(sr),'',
      '### 匹配对照配对空间组区间','',md(primary),'',
      f"新空间组跨重复100bp差{newcross.delta:+.6f}，95%组区间[{newcross.low:+.6f},{newcross.high:+.6f}]；历史块差{historicalcross.delta:+.6f}。新组同重复50bp差{same50.delta:+.6f}。必须同时考察PSNR/SSIM和对训练先验的比较，不能把单个形态数字的提高等同细节恢复。",'',
      '### 下游已知结构与噪声诊断','',
      f"{checks['known_unique_structures']}个实际完整的已知注释进入局部形态/信号峰位诊断，全部方法均评价。known_structure_per_window.csv保留逐例结果，known_structure_paired_intervals.csv列出空间组配对区间。10.24kb输入不足以运行要求完整12.8/25.6kb上下文的任务一分类器；本轮不补零冒充分类准确率，原49个全上下文下游基线仍保留。",'',
      'known_examples/按基因组坐标固定选取每类第一个合格实例，图中青线是原注释边界，并非模型预测边界；不按模型改善选图。另保留两个固定随机空间探针。可用scripts/reconstruct_shared_same.py --input-low <40bp均值.npy> --output <新输出.npz>直接应用验证选择的分支，或用--method allocation_shared_same应用本轮网络对照；应用示例与冻结推理逐像素核对。','',
      '噪声表保留全部241训练/验证窗、全部距离/计数层（含空层）；每个原生唯一40bp块的双重复总量、分配MSE/余弦值保存在可重建本地缓存及哈希清单。20个验证方向的直接Cooler聚合与native均值检查通过。跨重复相关只是经验诊断，不是理论信息上限。','',
      '下表相关系数为各窗口相应层内相关的算术平均，未定义者不伪填0，并同时列出有效窗口数与全部块数；不将不同层/不同样本量的相关混作全局相关。','',md(noise_summary),'',
      'repeat_error_decomposition.csv验证精确代数恒等式：两个重复目标的平均MSE=对两重复均值的MSE+两重复差平方/4。后项含生物差异和采样误差，两个重复不足以把它们分离，也不足以断言不可预测噪声的严格比例。','',
      '### 最终保留与停止条件','',
      f"共享训练相对匹配单源的跨重复100bp组区间在两套历史空间组均为正：{shared_morphology_benefit}。验证选中方案为{selection['selected_method']}，但是否替换正式默认还需要全部关键指标和局部结构取舍。原choices.json保持冻结；本轮停止网络结构/损失搜索，不以同一历史测试结果继续改权重。",'',
      '## 实现、运行、验证与待完成','',
      '已实现并运行：旧AE权重复现、支持掩码背景AE/固定方向联合/随机对照、留类新颖性、全部候选跨重复/聚类/条件核验、四个匹配SR模型、全窗口噪声诊断、统一验证与同/跨双方向评价。','',
      '已验证：原有工作哈希不变、来源归档哈希、相同分区与旧形态基线复现、原始聚合数学语义、新SR计数/对称/非负、全候选覆盖。单元测试及真实退出状态见runs/与run_status.csv。AE工程加速中止保留原模型后重读，并按所需索引逐值核验；没有按测试成绩改变方法。','',
      '待完成科学证据：外部/更深覆盖匹配重复、独立盲法候选复核、能区分嵌套结构类型的新颖性任务定义。本轮实验已经收尾，不代表新生物类型或可靠10bp细节恢复已解决。','',
      '运行入口：设置PYTHONPATH=src，使用项目Python运行scripts/run_targeted_review.py --stage <阶段>。配置、原数据缓存及第五轮比较权重为依赖；日志不可覆盖，复跑需新output目录。noise/conditions的大缓存本地保留，来源及SHA256已记录，可按脚本恢复。','',
      '修改文件清单见change_manifest.csv，运行状态见run_status.csv；完整CSV保留不利指标及淘汰理由。下一步优先投入外部数据和标签核验，不再以增加同类实验数量替代新证据。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('Actual-output report, paired comparisons and integrity checks generated',flush=True)

if __name__=='__main__':main()
