"""Geometry, anonymous evidence identities and append-only human review records."""
from datetime import datetime,timezone
import hashlib
import hmac
import json
from pathlib import Path
import numpy as np

def anonymous_id(center,secret):
    return 'BR-'+hmac.new(bytes.fromhex(secret),str(int(center)).encode(),hashlib.sha256).hexdigest()[:16]

def context_relation(center,start,end,annotation_starts,annotation_ends,buffer_bp):
    a,b=np.asarray(annotation_starts),np.asarray(annotation_ends)
    if ((a<=center)&(b>center)).any():return 'center_inside_known_interval'
    if ((a>=start)&(b<=end)).any():return 'input_contains_known_interval'
    if ((a<end+buffer_bp)&(b>start-buffer_bp)).any():return 'known_context_or_neighbor'
    return 'outside_known_context'

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def append_review(path,record,case_ids,categories,packet_sha256):
    required=['case_id','reviewer','assessment','evidence']
    if any(not str(record.get(key,'')).strip() for key in required):raise ValueError('Reviewer, case, assessment and evidence are required')
    if record['case_id'] not in case_ids or record['assessment'] not in categories:raise ValueError('Unknown case or assessment')
    path=Path(path);previous='0'*64;records=[]
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            item=json.loads(line);expected=hashlib.sha256(previous.encode()+canonical(item['record'])).hexdigest()
            if item['previous']!=previous or item['hash']!=expected:raise ValueError('Review history hash chain failed')
            records.append(item);previous=item['hash']
    value=dict(record);value.update(reviewed_utc=datetime.now(timezone.utc).isoformat(),packet_sha256=packet_sha256)
    item=dict(previous=previous,record=value,hash=hashlib.sha256(previous.encode()+canonical(value)).hexdigest())
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8',newline='\n') as out:out.write(json.dumps(item,ensure_ascii=False)+'\n')
    return item

def block_statistics(prediction,target,source,unique):
    """Unique native pixels within coarse blocks; zero-target errors remain included."""
    error=prediction-target;den=unique.sum((-2,-1))
    mae=(abs(error)*unique).sum((-2,-1))/den
    mse=(error**2*unique).sum((-2,-1))/den
    total=(target*unique).sum((-2,-1))
    pmass=(prediction*unique).sum((-2,-1));q=prediction/pmass[...,None,None].clip(1e-12);y=target/total[...,None,None].clip(1e-12)
    proportion=((q-y)**2*unique).sum((-2,-1))/den
    return dict(raw_mae=mae,raw_mse=mse,target_squared_mean=(target**2*unique).sum((-2,-1))/den,conditional_proportion_mse=proportion,common_positive=(source.sum((-2,-1))>0)&(total>0),target_zero=total==0)
