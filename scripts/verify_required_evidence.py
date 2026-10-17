"""Keep the blind packet frozen; validate real condition signals and targeted track examples."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
import requests
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.method_review import load_config,write_json,sha256
from dlpro.experiment_core import LENGTH,LABELS,ContactReader

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'required_evidence';out.mkdir(exist_ok=True);assert not (out/'status.json').exists();previous=Path(cfg['previous_output']);blind=previous/'blind_review';before={name:sha256(blind/name) for name in ['blind_review_packet.zip','public/index.html','human_review_status.json','selection_freeze.json']}
    public=json.loads((blind/'public/packet_manifest.json').read_text(encoding='utf-8'));human=json.loads((blind/'human_review_status.json').read_text(encoding='utf-8'));assert human['reviewed']==0
    with zipfile.ZipFile(blind/'blind_review_packet.zip') as archive:
        assert archive.testzip() is None;names=set(archive.namelist());assert not any(any(word in name for word in ['private','mapping','identity_key','ranking']) for name in names)
        for case in public['cases']:
            for key in ['image','data']:assert hashlib.sha256(archive.read(case[key])).hexdigest()==case[key+'_sha256']
    # End-to-end import/append/disagreement check in a disposable directory only.
    with tempfile.TemporaryDirectory(prefix='dlpro-review-engineering-') as temporary:
        tmp=Path(temporary);testout=tmp/'isolated';(testout/'blind_review/public').mkdir(parents=True);shutil.copy2(blind/'public/packet_manifest.json',testout/'blind_review/public/packet_manifest.json')
        testcfg=load_config('configs/evidence_review_v1.json');testcfg['output']=str(testout);write_json(tmp/'config.json',testcfg);case=public['cases'][0]['case_id']
        for index,(reviewer,assessment) in enumerate([('ENGINEERING_ONLY_A','known_form'),('ENGINEERING_ONLY_B','insufficient_evidence'),('ENGINEERING_ONLY_A','possible_nested_form')]):
            write_json(tmp/f'import{index}.json',[dict(case_id=case,reviewer=reviewer,assessment=assessment,evidence='Disposable integration fixture, not a human judgement')]);run=subprocess.run([sys.executable,'scripts/review_blind_candidates.py','--config',str(tmp/'config.json'),'--import-json',str(tmp/f'import{index}.json')],capture_output=True,text=True,encoding='utf-8');assert run.returncode==0,run.stderr
        status=json.loads((testout/'blind_review/human_review_status.json').read_text(encoding='utf-8'));checked=next(r for r in status['case_status'] if r['case_id']==case);assert checked['reviewers']==2 and checked['disagreement'];assert len((testout/'blind_review/human_reviews.jsonl').read_text(encoding='utf-8').splitlines())==3
    assert before=={name:sha256(blind/name) for name in before};assert not (blind/'human_reviews.jsonl').exists()
    instructions='''# 独立候选复核操作说明

适用对象：未参与本项目模型、阈值或候选选择的真实复核者。请先记录姓名或稳定人员代号、相关经验、是否接触过算法得分/坐标，以及可能的利益或先验影响。未能证明独立性时按普通复核记录，不标为独立验证。

## 获取与逐例查看

仅领取第七轮公开ZIP或其中public文件；不要领取private、坐标映射、完整研究报告、算法名、分数与排名。解压后打开index.html。按匿名ID逐例检查分配清单，全部分配案例都须处理；不能只挑明显或成功案例。尚无真实人员时项目状态保持待审。

检查WT1/WT2的相似形态、有效像素缺失、匹配普通背景及已知注释的相对上下文。可下载NPZ看真实100bp粗信号与掩码；它不是新增10bp读段真值。相同区域两个重复一致，也不自动等于独立于已知结构的新形态。

## 统一判定类别

| 网页类别 | 使用说明 |
|---|---|
| known_form | 可由已有已知类型解释，注明依据 |
| possible_boundary_change | 疑似已知结构的边界差异；不直接认定新类 |
| possible_nested_form | 疑似嵌套形态，注明包含/被包含关系及重复证据 |
| adjacent_form | 邻近既有结构，注明能否分离边界 |
| background_variation | 与背景或缺失/噪声解释相容 |
| distinct_repeatable_candidate | 两重复有区别于已知形态的独特候选；仍不是正式新簇确认 |
| insufficient_evidence | 无法可靠判断，写出缺少的证据 |

网页初始下拉框已有默认项，务必主动选择真实判断，不能将默认项当作完成。逐例填写证据、不确定性、有效像素是否妨碍判定、与背景的差异。先独立判断，再讨论分歧。不要用算法热图评分代替判断，不得用模型自动生成标签补齐未看案例。

## 保存、导出与分歧

填写复核人、判定、依据后点击保存本例。浏览器记录保存在本机，导出JSON并保留导出时间；也可填写公开review_template.csv。多人分别导出，保留各自原始文件。由项目人员验证身份/独立性后使用既有导入脚本导入真实记录，追加日志保留修订与分歧。实际导入UTC与复核者客户端时间分别保留；不能把导入时间当作实际阅图开始时间。所有正式复核只能在真实人员完成后统计。

复核结论与算法评分分开保存，不据同一批复核标签反复调分后又宣称独立验证。若以后用于开发，先制定开发/独立评估隔离。当前用户确认没有复核者，实际状态为0/243；本轮临时导入测试已销毁，不是人工审核。
'''
    (out/'INDEPENDENT_REVIEW_GUIDE.md').write_text(instructions,encoding='utf-8')
    sources=[]
    def restore(name):
        local=out/'committed_assets'/name
        if not local.exists():
            original=Path(name)
            if original.exists():payload=original.read_bytes()
            else:
                response=requests.get(f"https://raw.githubusercontent.com/ysaaa-rain/dl-pro/{cfg['source_revision']}/{name}",timeout=cfg['timeout_seconds']);response.raise_for_status();payload=response.content
            assert len(payload)<=cfg['maximum_small_asset_bytes'];local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(payload)
        payload=local.read_bytes();oid=subprocess.check_output(['git','rev-parse',f'HEAD:{name}'],text=True).strip();assert hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()==oid;sources.append(dict(original_path=name,local_path=local.as_posix(),sha256=sha256(local),git_blob=oid));return local
    samples=pd.read_csv('data/annotations/samples.csv');genes=pd.read_csv('data/annotations/genes.csv');structures=pd.read_csv('data/annotations/structures.csv');selected=[];signal_checks=[];track_checks=[];raw_checks=[];cache_by_sample={}
    for directory in ['multitrack','multitrack_mutants']:
        base='results/'+directory;index=pd.read_csv(restore(base+'/index.csv'));protocol=json.loads(restore(base+'/protocol.json').read_text(encoding='utf-8'));depth=pd.read_csv(restore(base+'/sample_depths.csv'));arrays=np.load(restore(base+'/signals.npz'));signals=pd.read_csv(restore(base+'/signals.csv.gz'));conditions=protocol['conditions'];assert len(set(conditions))>=3
        assert index.start.iloc[0]==0 and index.end.iloc[-1]==LENGTH and np.array_equal(index.start.to_numpy()[1:],index.end.to_numpy()[:-1]);assert (index.end-index.start).iloc[:-1].eq(protocol['interval_bp']).all()
        sample_subset=samples[samples.condition.isin(conditions)];assert sample_subset.groupby('condition').size().reindex(conditions).eq(2).all();assert np.array_equal(signals.start,np.arange(len(signals))*100) and signals.end.iloc[-1]==LENGTH
        for condition in conditions:
            subset=sample_subset[sample_subset.condition==condition];expected=np.mean([arrays[r.sample_id] for r in subset.itertuples()],axis=0);np.testing.assert_allclose(signals[condition],expected,rtol=1e-6,atol=1e-8);assert np.isfinite(expected).all() and (expected>=0).all();signal_checks.append(dict(directory=directory,condition=condition,repeats=len(subset),bins=len(expected),mean_merged_correct=True))
            for sample in subset.itertuples():
                assert sample.sample_id in arrays.files;cache_by_sample[sample.sample_id]=dict(values=arrays[sample.sample_id].copy(),path=sample.path,library_pairs=float(depth.set_index('sample_id').loc[sample.sample_id,'library_pairs']),sha256=sample.sha256,condition=condition)
        # Deterministic semantic coverage; never select by whether a model succeeds.
        choices=[('first',0),('middle',len(index)//2),('last_short_bin',len(index)-1)]
        gene_counts=[int(((genes.start<r.end)&(genes.end>r.start)).sum()) for r in index.itertuples()];choices.append(('most_genes',int(np.argmax(gene_counts))))
        for label in LABELS:
            known=structures[structures.type==label];counts=[int(((known.start<r.end)&(known.end>r.start)).sum()) for r in index.itertuples()];choices.append(('most_'+label,int(np.argmax(counts))))
        empty=np.flatnonzero(index.structures.to_numpy()==0);choices.append(('background_middle',int(empty[np.argmin(abs(empty-len(index)//2))])))
        reasons={}
        for reason,i in choices:reasons.setdefault(i,[]).append(reason)
        for i,why in sorted(reasons.items()):
            row=index.iloc[i];path=restore(str(row.path));assert gene_counts[i]==row.genes;assert int(((structures.start<row.end)&(structures.end>row.start)).sum())==row.structures
            with Image.open(path) as im:values=np.asarray(im.convert('RGB'));width,height=im.size;assert values.std()>1 and width>=600 and height>=400
            selected.append(dict(directory=directory,start=int(row.start),end=int(row.end),reason=','.join(why),local_path=path.as_posix(),genes=int(row.genes),structures=int(row.structures),width=width,height=height,image_not_blank=True))
        track_checks.append(dict(directory=directory,teacher='p10',conditions=conditions,n_figures=len(index),continuous_coverage=True,genome_bp=LENGTH,four_panels_declared=protocol['panels'],input_bin_bp=protocol['bin_bp'],missing_local_full_images='Sparse checkout: pinned Git contains full set; only targeted examples materialized',all_images_visually_reviewed=False))
    # A small direct raw-data check, no complete signal regeneration.
    for sample_id,item in cache_by_sample.items():
        reader=ContactReader(item['path']);assert np.isclose(reader.library_size,item['library_pairs'])
        for position in [0,len(item['values'])//2,len(item['values'])-1]:
            matrix=reader.window(position-100,n=201,bin_bp=100);value=float(matrix[100].sum()-matrix[100,100]);expected=float(item['values'][position]);assert np.isclose(value,expected,rtol=1e-5,atol=1e-7),(sample_id,position,value,expected)
            raw_checks.append(dict(sample_id=sample_id,condition=item['condition'],coarse_bin=position,signal_from_raw=value,saved_signal=expected,absolute_error=abs(value-expected),source_sha256_from_provenance=item['sha256'],passed=True))
        print('Raw track signal verified',sample_id,flush=True)
    pd.DataFrame(selected).to_csv(out/'selected_track_images.csv',index=False);pd.DataFrame(signal_checks).to_csv(out/'merged_signal_checks.csv',index=False);pd.DataFrame(raw_checks).to_csv(out/'raw_signal_checks.csv',index=False);write_json(out/'track_requirements.json',track_checks);write_json(out/'source_manifest.json',sources)
    # Read-only contact sheets support actual visual checking of exactly these examples.
    for number,start in enumerate(range(0,len(selected),4)):
        fig,axes=plt.subplots(2,2,figsize=(18,12),constrained_layout=True)
        for ax,item in zip(axes.flat,selected[start:start+4]):ax.imshow(plt.imread(item['local_path']));ax.set_title(item['directory']+' '+item['reason']);ax.axis('off')
        for ax in list(axes.flat)[len(selected[start:start+4]):]:ax.axis('off')
        fig.savefig(out/f'track_review_sheet_{number+1}.png',dpi=110);plt.close(fig)
    write_json(out/'status.json',dict(blind_packet_unchanged=True,real_reviewed_cases=human['reviewed'],total_cases=len(public['cases']),reviewer_available=False,engineering_import_disagreement_revision_test=True,engineering_records_disposed=True,task3_condition_sets=len(track_checks),task3_targeted_images=len(selected),task3_raw_signal_checks=len(raw_checks),task3_signal_merge_verified=True,task3_all_images_visual_checked=False,task5_formal_choice_unchanged=True))
    print('Frozen blind packet, independent instructions and targeted required tracks verified',flush=True)

if __name__=='__main__':main()
