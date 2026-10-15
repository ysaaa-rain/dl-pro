"""Sequential continuation after SR selection; halt on any failed stage."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
selection=root/'results/sr_data_controls/selection.json'
wait_for=root/'results/sr_data_controls/evaluation_protocol.json' if '--after-sr-evaluation' in sys.argv else selection
while not wait_for.exists():time.sleep(5)
steps=['train_classifier_repeat_ssl.py','evaluate_sr_data_controls.py','evaluate_classifier_repeat_ssl.py','summarize_performance_controls.py','audit_performance_controls.py','plot_performance_controls.py'];state=[]
if '--after-sr-evaluation' in sys.argv:steps.remove('evaluate_sr_data_controls.py')
for script in steps:
 entry=dict(script=script,status='running',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));state.append(entry);(root/'results/performance_protocol/pipeline_status.json').write_text(json.dumps(state,indent=2)+'\n');print('START',script,flush=True)
 with (root/'results/experiment'/('performance_'+script.replace('.py','.log'))).open('w') as log:result=subprocess.run([sys.executable,str(root/'scripts'/script)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 entry.update(status='complete' if result.returncode==0 else 'failed',ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),returncode=result.returncode);(root/'results/performance_protocol/pipeline_status.json').write_text(json.dumps(state,indent=2)+'\n')
 if result.returncode:raise SystemExit(result.returncode)
 print('COMPLETE',script,flush=True)
