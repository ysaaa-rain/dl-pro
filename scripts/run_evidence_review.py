"""Seventh-round stages; no implicit training or changes to frozen decisions."""
import argparse
from dlpro.method_review import load_config,logged_run

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/evidence_review_v1.json')
    p.add_argument('--stage',required=True,choices=['audit','metadata','blind','novelty','superresolution','materials','report','test']);p.add_argument('--attempt',default='')
    a=p.parse_args();cfg=load_config(a.config)
    scripts=dict(audit='audit_course_evidence.py',metadata='audit_independent_metadata.py',blind='prepare_blind_evidence.py',novelty='diagnose_novelty_evidence.py',superresolution='audit_sr_evidence.py',materials='verify_evidence_materials.py',report='summarize_evidence_review.py')
    command=['-m','pytest','-q','tests/test_evidence_review.py'] if a.stage=='test' else ['scripts/'+scripts[a.stage],'--config',a.config]
    logged_run(cfg,a.stage+a.attempt,command)

if __name__=='__main__':main()
