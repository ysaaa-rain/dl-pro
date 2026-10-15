"""Read-only/export/error-path checks against a running local annotation server."""
import json,re
from urllib.request import urlopen,Request
from urllib.error import HTTPError
from dlpro.experiment_core import save_json
base='http://127.0.0.1:8765';html=urlopen(base).read().decode();token=re.search("'X-Session-Token':'([^']+)'",html).group(1)
def post(path,data,session=token):
 request=Request(base+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json','X-Session-Token':session})
 try:
  with urlopen(request) as response:return response.status,json.load(response)
 except HTTPError as e:return e.code,json.load(e)
assert post('/api/predict',{'center':1},'invalid')[0]==403
assert post('/api/predict',{'center':-1})[0]==400
assert post('/api/predict',{'center':True})[0]==400
assert post('/api/predict',[])[0]==400
assert post('/api/review',{'prediction_id':'missing','decision':'CHID'})[0]==400
code,pred=post('/api/predict',{'center':153487});assert code==200 and pred['input_source']=='known_cache';assert len(pred['view_intervals'])==3
exported=urlopen(base+'/api/export').read().decode('utf-8-sig');assert 'human_label' in exported
save_json('results/annotation_tool/http_checks.json',dict(forbidden_session=403,invalid_coordinate=400,boolean_coordinate=400,non_object_json=400,missing_prediction=400,known_cache_prediction=200,csv_export=200,ui_save_restart_and_download='Observed in browser with isolated.qa test database; no real annotation synthesized'))
print('HTTP checks passed')
