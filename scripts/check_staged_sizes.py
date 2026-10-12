"""Fail before commit if any staged blob exceeds GitHub's 100 MiB limit.

No dataset is ignored. Large originals remain local and are listed by SHA256.
Run: python scripts/check_staged_sizes.py
"""
import subprocess
import sys

records = subprocess.check_output(['git', 'ls-files', '-s', '-z']).split(b'\0')
bad=[]
for record in records:
    if not record: continue
    meta,path=record.split(b'\t',1)
    sha=meta.split()[1].decode()
    size=int(subprocess.check_output(['git','cat-file','-s',sha]))
    if size >= 100*1024**2:
        bad.append((path.decode(),size))
for path,size in bad: print(f'Large staged blob: {size} bytes {path}')
sys.exit(bool(bad))
