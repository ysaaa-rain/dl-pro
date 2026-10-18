"""Evidence assessments for retained candidates; no cluster-level admission gates."""
import numpy as np


def assess_inventory(frame):
    frame=frame.copy()
    # This flag concerns individual evidence, never whether a row is output.
    frame['tentative_new_shape']=(frame.quality_pass & ~frame.known_buffer_overlap
        & frame.both_repeats_beyond_known_distance
        & (frame.top3_geometry_difference_directions_agree>=2)
        & (frame.matched_background_count>=20)
        & (frame.matched_background_localization_tail<=.05)
        & (frame.local_repeat_boundary_iou>=.5)
        & (frame.minimum_cross_scale_interval_iou>=.5))
    reasons=[]
    for row in frame.itertuples():
        limitations=[]
        if not row.quality_pass:limitations.append('基础有效像素或支持像素上的重复相关不足')
        if row.known_buffer_overlap:limitations.append('扫描区间或200 bp缓冲区与已知注释关联')
        elif row.known_full_context_overlap:limitations.append('较大上下文仍含已知注释，存在邻近信号解释')
        if not row.both_repeats_beyond_known_distance:limitations.append('整窗几何未同时超出已知参考距离范围；可能接近未标注的已知形态')
        if row.top3_geometry_difference_directions_agree<2:limitations.append('主要几何差异的跨重复方向不足')
        if row.matched_background_count<20:limitations.append('可比背景数量不足20')
        if not np.isfinite(row.matched_background_localization_tail) or row.matched_background_localization_tail>.05:
            limitations.append('局部定位分数未显示足够的匹配背景罕见性')
        if row.local_repeat_boundary_iou<.5:limitations.append('两重复的局部定位区域不一致')
        if row.minimum_cross_scale_interval_iou<.5:limitations.append('跨尺度可能定位到不同对象')
        if row.tentative_new_shape:limitations.append('具备探索性新形态信号，仍需独立验证和人工形态复核')
        reasons.append('；'.join(limitations))
    frame['novelty_evidence_assessment']=reasons
    frame['analysis_admission']='retained regardless of cluster size or known-member composition'
    return frame
