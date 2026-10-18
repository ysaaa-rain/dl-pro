"""Read original course material, verify round-six artifacts and grade gaps first."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import requests
import fitz
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from dlpro.method_review import load_config,sha256,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'course_audit';out.mkdir(exist_ok=True)
    assert not (out/'status.json').exists()
    git={name:subprocess.check_output(command,text=True,encoding='utf-8').strip() for name,command in dict(branch=['git','branch','--show-current'],head=['git','rev-parse','HEAD'],status=['git','-c','core.quotepath=false','status','--short','--branch'],stashes=['git','stash','list'],staged=['git','diff','--cached','--name-only'],remotes=['git','remote','-v']).items()}
    write_json(out/'initial_git_state.json',git)
    protected={}
    for base in [Path(cfg['previous_output']),Path(cfg['sixth_output'])]:
        manifest=base/'review_final/change_manifest.csv' if base==Path(cfg['previous_output']) else base/'review/change_manifest.csv'
        for row in csv.DictReader(manifest.open(encoding='utf-8')):
            assert Path(row['path']).is_file() and sha256(row['path'])==row['sha256'],row['path']
            protected[row['path']]=row['sha256']
        protected[manifest.as_posix()]=sha256(manifest)
    # Capture all pre-existing source/config/test files too, not just last-round changes.
    for directory,pattern in [('src','*.py'),('scripts','*.py'),('tests','*.py'),('configs','*.json')]:
        for path in Path(directory).rglob(pattern):
            if 'evidence' in path.name or path.name in ['prepare_blind_evidence.py','audit_independent_metadata.py']:continue
            protected[path.as_posix()]=sha256(path)
    protected['results/experiment_choices/choices.json']=sha256('results/experiment_choices/choices.json')
    write_json(root/'protected_existing.json',protected)
    doc=fitz.open(cfg['teacher_pdf']);pages=[]
    for i,page in enumerate(doc):pages.append(dict(page=i+1,text=page.get_text()))
    write_json(out/'teacher_pages.json',dict(path=cfg['teacher_pdf'],sha256=sha256(cfg['teacher_pdf']),pages=pages))
    for page in [9,10,12,14]:doc[page-1].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(out/f'teacher_page_{page}.png')
    # Restore only small, committed evidence that the sparse checkout lacks; verify Git blob IDs.
    wanted=['results/classification_fine/fine_consistent_ensemble_report.json','results/classification_fine/explanations/protocol.json','results/classification_fine/explanations/occlusion_summary.csv','results/multitrack/index.csv','results/multitrack/protocol.json','results/multitrack_mutants/index.csv','results/multitrack_mutants/protocol.json','results/experiment/reference_resolution.json']
    evidence=[];tree=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],text=True,encoding='utf-8').splitlines();tree_set=set(tree)
    for name in wanted:
        if name not in tree_set:continue
        if Path(name).exists():payload=Path(name).read_bytes();location=Path(name)
        else:
            response=requests.get(f"https://raw.githubusercontent.com/ysaaa-rain/dl-pro/{git['head']}/{name}",timeout=cfg['evidence']['network_timeout_seconds']);response.raise_for_status();payload=response.content
            location=out/'committed_evidence'/name;location.parent.mkdir(parents=True,exist_ok=True);location.write_bytes(payload)
        oid=subprocess.check_output(['git','rev-parse',f"HEAD:{name}"],text=True).strip();actual=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
        assert actual==oid,name
        evidence.append(dict(original_path=name,local_path=location.as_posix(),git_blob=oid,sha256=hashlib.sha256(payload).hexdigest()))
    write_json(out/'committed_evidence_manifest.json',evidence)
    frame=pd.read_csv('results/classification_fine/out_of_fold_predictions.csv');choices=json.loads(Path('results/experiment_choices/choices.json').read_text(encoding='utf-8'))
    recipe=choices['classification']['primary'];part=frame[frame.recipe==recipe];assert part.structure_id.nunique()==len(part)>0
    classification=dict(n=len(part),groups=int(part.group.nunique()),recipe=recipe,accuracy=accuracy_score(part.true_type,part.predicted_type),macro_f1=f1_score(part.true_type,part.predicted_type,average='macro'),report=classification_report(part.true_type,part.predicted_type,output_dict=True),confusion=confusion_matrix(part.true_type,part.predicted_type,labels=['CHID','CHIN','OPCID']).tolist())
    write_json(out/'classification_verified.json',classification)
    tracks=[]
    for directory in sorted({p.split('/')[1] for p in tree if p.startswith('results/multitrack')}):
        images=[p for p in tree if p.startswith(f'results/{directory}/tracks/') and p.endswith('.png')]
        item=dict(directory=directory,committed_png_count=len(images),local_png_count=len(list(Path('results',directory,'tracks').glob('*.png'))))
        manifest=next((v for v in evidence if v['original_path']==f'results/{directory}/index.csv'),None)
        if manifest:
            index=pd.read_csv(manifest['local_path']);item['index_columns']=index.columns.tolist();item['index_rows']=len(index)
            if {'start','end'}.issubset(index):
                order=index.sort_values('start');item['continuous_genome_coverage']=bool(order.start.iloc[0]==0 and (order.start.to_numpy()[1:]==order.end.to_numpy()[:-1]).all() and order.end.iloc[-1]==4641652)
        tracks.append(item)
    write_json(out/'track_inventory.json',tracks)
    sixth=Path(cfg['sixth_output']);actual=pd.read_csv(sixth/'evaluation/summary.csv')
    quoted=actual[(actual.split=='new_test')&(actual.comparison=='cross_repeat')&actual.method.isin(['train_distance_prior','bicubic_unprojected'])]
    quoted.to_csv(out/'sixth_claims_verified.csv',index=False)
    choices_hash=sha256('results/experiment_choices/choices.json')
    write_json(out/'frozen_choices.json',dict(formal_choices=choices,formal_choices_sha256=choices_hash,sixth_decisions=json.loads((sixth/'review/decisions.json').read_text(encoding='utf-8')),sixth_test_lock_sha256=sha256(sixth/'evaluation/test_lock.json')))
    # These are assessment categories, not fabricated points or teacher score predictions.
    rows=[
      ['任务一40分：逐类可靠识别','p8-9','classification_verified.json；344条、56组折外预测','部分满足','必做高权重；CHID召回不足，不能以整体准确率代替逐类稳定','保留分类型混淆、样本量和置信边界，停止按历史测试挑模型'],
      ['任务一：权重、训练评估脚本及准确率/混淆矩阵','p9','15权重及5变换；原OOF重新汇总','已满足且有验证证据','可复现与实现评分','整理版本和数据依赖'],
      ['任务一：显著性须关注结构而非背景','p9','已提交梯度与遮挡；小体积协议/遮挡摘要按Git对象核验','已实现但验证不足','解释性15%的一部分；图存在不等于关注结构证明','报告遮挡数值与局限，必要时补盲法解释复核'],
      ['任务二30分：全基因组候选与已知召回','p9-10','原726窗口、344已知复现；第六轮243合格已知与540格点严格子集','已满足且有验证证据','满足最低流程标准；子集结果不能冒充全基因组','复用历史全覆盖和新空间子集，分别列出'],
      ['任务二：固定维度形状表征能区分背景','p9','形态、PCA和AE编码器；已有留类/背景评价','部分满足','方法设计与实验评价；不同类别差异大','按类别/折诊断表征与噪声混杂'],
      ['任务二：分组并标记已知/疑似新簇','p9','固定KMeans/HDBSCAN及已知上下文审计','已满足且有验证证据','工程完成；0合格簇不自动否定课程最低完成','保留0结果与判定依据，不降低阈值'],
      ['任务二：真正新颖性与独立复核','p9-10','两个已使用WT重复，多条件描述；无新簇和独立人员复核','尚未满足','科学证据不足；不等于课程最低流程未完成','生成全候选盲法包，核验外部数据/标签'],
      ['任务二：每候选CSV及统一热图','p10','第六轮2682排序记录、243唯一中心、243图、11样本','已满足且有验证证据','解释与展示及可复现','匿名化完整清单，评分与复核分离'],
      ['任务三10分：至少三条件四轨与全基因组覆盖','p10','Git提交的轨道索引/协议/图片清单，稀疏检出不包含全图','部分满足','必做任务不能因只看2/5忽略；需区分提交存在与本地视觉复核','核验连续区间、轨道定义与少量代表图；不声称全图视觉审查'],
      ['任务五选做20分：合法配对、模型权重和重建输出','p12','40bp均值→10bp；14网络权重、冻结评估与推理示例','已满足且有验证证据','实现及可复现已完成','固定输入语义与全部强基线'],
      ['任务五：明显优于插值并稳定恢复结构','p12','第六轮统一结果：计数投影取舍，多尺度无一致优势；19局部注释','部分满足','主要科学完成标准尚不能称充分达成','同时比较未投影/投影插值，分层实用收益与下游限制'],
      ['提交：代码ZIP、PPT讲解视频、报告分别提交','p14','正式成果按用户要求暂停；现有材料不作为最终验收','尚未满足','最终交付硬要求；不因实验多自动满足','本轮仅列清单，待用户授权再整理正式成果'],
      ['报告含设计思路、对照、指标、表图与可操作新结构定义','p14','实验Markdown与真实数据完整；正式报告暂停','部分满足','方法设计25%+实验评价25%；可转化证据已有','建立论证链并写明反例，正式文档保持暂停'],
      ['组号/姓名/学号/成员贡献百分比与命名','p14','尚无经用户确认的完整成员信息','因缺少资料无法判断','提交规范；不能编造','仅列缺项，不擅填'],
      ['加分：标注工具/清晰展示/LLM辅助','p9,p14','既有辅助标注工具；未验证人工效率，LLM辅助未实现','部分满足','加分不代替必做；不保证具体分数','本轮完成可审计的盲法工具，不冒称独立审核完成']]
    columns=['评分要求','教师原始依据','当前证据','状态','对得分的影响','下一步行动'];pd.DataFrame(rows,columns=columns).to_csv(out/'grading_gap.csv',index=False)
    priorities=[dict(rank=1,work='全候选匿名复核工具与证据包',grading_value='高：实验评价/展示',scientific_value='高：补人工独立证据接口',cost='中',success_probability='材料可完成；实际独立审核需人员'),dict(rank=2,work='强插值与约束取舍的统一SR审计',grading_value='高：选做主要完成标准',scientific_value='高：可证伪现有优势',cost='低到中',success_probability='审计可完成；性能提升不保证'),dict(rank=3,work='独立数据/作者标签与嵌套定义核验',grading_value='高：问题定义与设计依据',scientific_value='高',cost='低：仅元数据/小标签',success_probability='兼容性可核验；新深测序数据未保证'),dict(rank=4,work='按类别/空间折与噪声分层定位失败',grading_value='高：实验评价论证',scientific_value='高',cost='中：复用权重与CSV',success_probability='可完成；无改进亦有结论'),dict(rank=5,work='任务一/三及最终提交缺口清单',grading_value='高：必做和提交',scientific_value='中',cost='低',success_probability='可完成审计；正式成果仍暂停')]
    write_json(out/'priorities.json',priorities)
    write_json(out/'status.json',dict(original_teacher_read=True,teacher_pages=len(pages),prior_manifest_files_verified=len(protected),grade_rows=len(rows),classification_verified=True,sixth_rows=len(pd.read_csv(sixth/'evaluation/per_window.csv')),formal_choice_not_distance_prior=choices['superresolution']['primary']!='train_distance_prior',new_training_started=False))
    print('Original course requirements, sixth-round evidence and prioritized grade gaps audited',flush=True)

if __name__=='__main__':main()
