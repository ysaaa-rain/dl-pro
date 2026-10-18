import argparse
from dlpro.method_review import load_config,logged_run
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/classification_review_v1.json');p.add_argument('--stage',required=True,choices=['audit','experiment','explain','evidence','report','plot','test']);p.add_argument('--attempt',default='');a=p.parse_args();cfg=load_config(a.config)
scripts=dict(audit='audit_classification_review.py',experiment='experiment_classification_review.py',explain='explain_classification_review.py',evidence='verify_required_evidence.py',report='report_classification_review.py',plot='plot_classification_review.py')
cmd=['-m','pytest','-q','tests/test_classification_review.py'] if a.stage=='test' else ['scripts/'+scripts[a.stage],'--config',a.config]
logged_run(cfg,a.stage+a.attempt,cmd)
