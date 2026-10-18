"""Immutable stage runner for the sixth, focused review; existing artifacts untouched."""
import argparse
from dlpro.method_review import load_config, logged_run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/targeted_review_v1.json')
    p.add_argument('--stage',choices=['audit','ae','train','noise','evaluate','candidate_audit','cluster_audit','sources','conditions','known_examples','report','test'],required=True)
    p.add_argument('--attempt',default='')
    a=p.parse_args();cfg=load_config(a.config)
    command = {
        'audit':['scripts/audit_targeted_review.py'],
        'ae':['scripts/background_ae_review.py'],
        'train':['scripts/train_shared_same.py'],
        'noise':['scripts/diagnose_shared_noise.py'],
        'evaluate':['scripts/evaluate_shared_same.py'],
        'candidate_audit':['scripts/audit_novelty_candidates.py'],
        'cluster_audit':['scripts/cluster_review_candidates.py'],
        'sources':['scripts/restore_targeted_conditions.py'],
        'conditions':['scripts/validate_all_targeted_candidates.py'],
        'known_examples':['scripts/plot_targeted_known_examples.py'],
        'report':['scripts/summarize_targeted_review.py'],
        'test':['-m','pytest','-q','tests/test_targeted_review.py'],
    }[a.stage]
    if a.stage!='test': command+=['--config',a.config]
    logged_run(cfg,a.stage+a.attempt,command)

if __name__=='__main__': main()
