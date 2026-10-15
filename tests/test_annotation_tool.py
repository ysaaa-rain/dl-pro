import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from dlpro.annotation_tool import AnnotationEngine,ReviewStore

ROOT=Path(__file__).resolve().parents[1]

def test_frozen_inference_matches_oof_probabilities():
    engine=AnnotationEngine(ROOT)
    predictions=pd.read_csv(ROOT/'results/classification_fine/out_of_fold_predictions.csv')
    predictions=predictions[predictions.recipe=='fine_consistent_ensemble']
    for fold in range(5):
        expected=predictions[predictions.fold==fold].iloc[0]
        center=int(engine.known.set_index('structure_id').loc[expected.structure_id,'center'])
        actual=engine.predict(center)
        assert actual['fold']==fold and actual['evaluation_safe']
        assert actual['input_source']=='known_cache'
        assert np.allclose(list(actual['probabilities'].values()),expected[['p_CHID','p_CHIN','p_OPCID']].to_numpy(dtype=float),atol=1e-6)
        assert np.asarray(actual['saliency']).shape==(2,3,128,128)
        assert np.isfinite(actual['saliency']).all()
    for invalid in [-1,4641652,True,1.5,None,'12']:
        with pytest.raises(ValueError):engine.predict(invalid)


def test_review_append_only_restart_and_safe_export(tmp_path):
    path=tmp_path/'reviews.sqlite3';store=ReviewStore(path)
    prediction=dict(center=123,predicted_type='CHIN',model_sha256='hash',fold=2,evaluation_safe=True,views=[1],saliency=[2])
    store.save(prediction,'CHID',"=1+1")
    store.save(prediction,'不确定',"引号'和逗号,\n换行<script>")
    restored=ReviewStore(path).records();assert len(restored)==2
    assert restored[0]['decision']=='不确定' and restored[1]['decision']=='CHID'
    assert 'views' not in restored[0]['prediction']
    exported=store.export_csv();assert "'=1+1" in exported and 'human_label' in exported
    with pytest.raises(ValueError):store.save(prediction,'bad','x')
    with pytest.raises(ValueError):store.save(prediction,'CHID','a'*2001)
    assert len(store.records())==2
