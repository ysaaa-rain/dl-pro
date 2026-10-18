"""Import real human reviews, preserving revisions and disagreements; no model updates."""
import argparse
import csv
import json
from pathlib import Path
from dlpro.method_review import load_config,sha256,write_json
from dlpro.evidence_review import append_review

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/evidence_review_v1.json');group=p.add_mutually_exclusive_group();group.add_argument('--import-csv',type=Path);group.add_argument('--import-json',type=Path);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output'])/'blind_review'
    manifest=json.loads((root/'public/packet_manifest.json').read_text(encoding='utf-8'));ids={c['case_id'] for c in manifest['cases']}
    if a.import_csv or a.import_json:
        imported=list(csv.DictReader(a.import_csv.open(encoding='utf-8-sig'))) if a.import_csv else json.loads(a.import_json.read_text(encoding='utf-8-sig'))
        for record in imported:
            append_review(root/'human_reviews.jsonl',record,ids,set(cfg['evidence']['review_categories']),sha256(root/'public/packet_manifest.json'))
    rows=[]
    if (root/'human_reviews.jsonl').exists():rows=[json.loads(line)['record'] for line in (root/'human_reviews.jsonl').read_text(encoding='utf-8').splitlines()]
    # Latest per reviewer/case counts; keep every previous event in the original log.
    latest={}
    for row in rows:latest[(row['case_id'],row['reviewer'])]=row
    coverage=[]
    for case in sorted(ids):
        reviews=[r for (c,_),r in latest.items() if c==case];labels={r['assessment'] for r in reviews}
        coverage.append(dict(case_id=case,reviewers=len(reviews),status='pending' if not reviews else 'reviewed',disagreement=len(labels)>1,assessments=sorted(labels)))
    write_json(root/'human_review_status.json',dict(cases=len(ids),reviewed=sum(x['status']=='reviewed' for x in coverage),independent_blind_validation_completed=False,independence='Reviewer independence must be documented externally; tool operation is not independent review',case_status=coverage))
    print('Human review events',len(rows),'reviewed cases',sum(x['status']=='reviewed' for x in coverage),flush=True)

if __name__=='__main__':main()
