"""Export unannotated structure candidates and their morphology evidence."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/discovery_candidate_analysis'
OUT = SOURCE / 'findings'
# Visual descriptions concern the supplied paired 100 bp heatmaps, not class labels.
INTERPRETATIONS = {
    'C0011': ('块状与条带复合信号', '保留：未注释形态异常候选', '重复中存在相似宽域信号；缺失支持行穿过局部区域，背景仅3个，跨尺度区间不一致，不能确定独立的新形态。'),
    'C0059': ('边缘条带与角部富集', '保留：邻近已知结构的形态异常候选', '两重复边缘条带一致；大上下文含已知结构，局部跨尺度位置不同，存在邻近结构或窗口裁切解释。'),
    'C0144': ('块状富集伴边界条带', '保留：重点形态异常候选', '两重复均有清晰块状富集与边界条带，且公共有效像素完整；跨尺度区间变化，背景仅13个。可作为代表性未注释结构候选，尚不能排除已知形态变体或复合形态。'),
    'C0186': ('边缘富集与宽域弱信号', '保留：形态归属未定候选', '局部区域靠近窗口边缘且穿过缺失支持行；重复整体相关通过基础标准，局部跨尺度位置不同，仅1个匹配背景，形态解释不确定。'),
    'C0216': ('条带及局部富集复合信号', '保留：背景区分较弱候选', '重复相关为六条最低；大上下文含已知结构，局部信号在匹配背景中常见，不能归因于独立新形态。'),
    'C0230': ('十字形条带伴中心富集', '保留：重点可复现局部结构候选', '两重复局部定位和跨尺度位置稳定；与CHIN_127参考均呈明显十字形条带，整窗几何差异不能排除CHIN样形态变体，背景定位尾比例0.156也不支持罕见性。'),
}


def main():
    OUT.mkdir(exist_ok=True)
    source = SOURCE / 'candidate_inventory.csv'
    frame = pd.read_csv(source)
    selected = frame.loc[
        frame.quality_pass
        & ~frame.known_buffer_overlap
        & frame.both_repeats_beyond_known_distance
        & frame.top3_geometry_difference_directions_agree.ge(2)
    ].copy()
    assert set(selected.candidate_id) == set(INTERPRETATIONS), 'Operational candidate set differs from documented interpretations'
    selected['observed_morphology'] = selected.candidate_id.map(lambda x: INTERPRETATIONS[x][0])
    selected['finding_conclusion'] = selected.candidate_id.map(lambda x: INTERPRETATIONS[x][1])
    selected['interpretation'] = selected.candidate_id.map(lambda x: INTERPRETATIONS[x][2])
    selected['necessary_candidate_conditions_pass'] = selected.quality_pass & ~selected.known_buffer_overlap
    selected['direct_disqualifying_evidence_found'] = ~selected.necessary_candidate_conditions_pass
    selected['retained_finding'] = ~selected.direct_disqualifying_evidence_found
    selected['operational_unannotated_shape_finding'] = (selected.necessary_candidate_conditions_pass
        & selected.both_repeats_beyond_known_distance
        & selected.top3_geometry_difference_directions_agree.ge(2))
    selected['confirmed_new_structure_type'] = False
    selected['heatmap_file'] = '../' + selected.heatmap_file
    selected.to_csv(OUT / 'unannotated_morphology_findings.csv', index=False)
    knownlike = frame[frame.evidence_category.eq('unannotated_known_like')].copy()
    knownlike['finding_conclusion'] = '未注释的已知样结构候选；参考相似性不是已确认的类别归属'
    knownlike['heatmap_file'] = '../' + knownlike.heatmap_file
    knownlike.to_csv(OUT / 'unannotated_known_like_findings.csv', index=False)
    for f in (selected, knownlike):
        for row in f.itertuples():
            assert (OUT / row.heatmap_file).is_file()
    assert len(selected) == 6 and len(knownlike) == 28
    assert selected.retained_finding.all()
    summary = dict(morphology_candidates=6, retained=6, directly_disqualified=0,
                   unannotated_known_like=28, confirmed_new_structure_types=0,
                   review_scope='Existing measurements and paired 100 bp visual review; no new matrix experiment',
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
