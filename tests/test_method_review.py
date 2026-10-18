import numpy as np
from dlpro.method_review import load_config, spatial_cv_masks


def test_unassigned_tail_never_becomes_default_training_data():
    folds = np.array([-1, 0, 1, 2, 3, 4, 5])
    for fold in range(5):
        train, validation, test = spatial_cv_masks(np.ones(7, bool), folds, fold, 5)
        assert not (train | validation | test)[[0, 6]].any()
        assert (train | validation | test)[1:6].all()
        assert not ((train & validation) | (train & test) | (validation & test)).any()


def test_composed_configs_preserve_exact_sr_protocol_and_share_base_parameters():
    corrected = load_config('configs/method_review_v1_corrected.json')
    final = load_config('configs/method_review_final.json')
    assert final['sr'] == corrected['sr']
    assert final['seed'] == corrected['seed']
    assert final['discovery']['global_budgets'] == corrected['discovery']['global_budgets']
