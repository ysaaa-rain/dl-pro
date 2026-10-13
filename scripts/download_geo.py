"""Resumable download of all GEO GSE272159 Cooler matrices in the manifest."""
import concurrent.futures
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

MANIFEST=Path('results/geo_download_manifest.json')
ROOT=Path('data/raw')
STATE=Path('results/geo_download_state.json')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def download(item):
    name=item['name']; target=ROOT/name; partial=ROOT/(name+'.part')
    expected=item['content_length']; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.stat().st_size==expected:
        with gzip.open(target,'rb') as f:
            while f.read(16*1024*1024): pass
        return name,expected,digest(target),'already-present-verified'
    if partial.exists() and partial.stat().st_size==expected:
        try:
            with gzip.open(partial,'rb') as f:
                while f.read(16*1024*1024): pass
            os.replace(partial,target)
            return name,expected,digest(target),'completed-part-verified'
        except Exception:
            partial.unlink()
    for attempt in range(8):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>=expected:
            if offset==expected:
                try:
                    with gzip.open(partial,'rb') as f:
                        while f.read(16*1024*1024): pass
                    os.replace(partial,target)
                    return name,expected,digest(target),'completed-part-verified'
                except Exception:
                    pass
            partial.unlink(); offset=0
        headers={'User-Agent':'dl-pro-course-experiment/1.0'}
        if offset: headers['Range']=f'bytes={offset}-'
        req=urllib.request.Request(item['url'],headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=120) as response:
                if offset and response.status!=206:
                    raise IOError(f'resumed response was HTTP {response.status}')
                if response.status not in (200,206): raise IOError(f'HTTP {response.status}')
                if offset:
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or int(match[1])!=offset or int(match[3])!=expected:
                        raise IOError('Unexpected Content-Range; refusing to append')
                mode='ab' if offset else 'wb'
                with partial.open(mode) as out:
                    while True:
                        chunk=response.read(4*1024*1024)
                        if not chunk: break
                        out.write(chunk)
                if partial.stat().st_size != expected:
                    raise IOError(f'got {partial.stat().st_size} of {expected} bytes')
                with gzip.open(partial,'rb') as f:
                    while f.read(16*1024*1024): pass
                os.replace(partial,target)
                return name,expected,digest(target),'downloaded-verified'
        except Exception:
            if attempt==7: raise
            time.sleep(min(5*(attempt+1),30))
    raise RuntimeError(name)


def main():
    manifest=json.loads(MANIFEST.read_text())
    items=manifest['files']; completed=[]; failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(download,x):x for x in items}
        for future in concurrent.futures.as_completed(futures):
            item=futures[future]
            try:
                result=future.result();completed.append(dict(zip(['name','bytes','sha256','status'],result)))
                print(f"Verified {result[0]} ({result[1]} bytes)",flush=True)
            except Exception as exc:
                failures.append({'name':item['name'],'error':repr(exc)})
                print(f"FAILED {item['name']}: {exc}",flush=True)
            tmp=STATE.with_suffix('.json.tmp')
            tmp.write_text(json.dumps({'expected_files':len(items),'completed':completed,'failures':failures},indent=2)+'\n')
            os.replace(tmp,STATE)
    if failures: raise SystemExit(f'{len(failures)} downloads failed; rerun to resume')

if __name__=='__main__':
    with Path('.git/dlpro-download.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another download process is active')
        main()
