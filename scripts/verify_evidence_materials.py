"""Exercise the offline packet and sample pinned course tracks, without human reviews."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
import pandas as pd
import requests
from PIL import Image
from playwright.sync_api import sync_playwright
from dlpro.method_review import load_config,sha256,write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=load_config(args.config);root=Path(cfg['output']);out=root/'material_verification';out.mkdir(exist_ok=True)
    assert not (out/'status.json').exists()
    public=root/'blind_review/public';manifest=json.loads((public/'packet_manifest.json').read_text(encoding='utf-8'))
    for case in manifest['cases']:
        assert sha256(public/case['image'])==case['image_sha256']
        assert sha256(public/case['data'])==case['data_sha256']
        with Image.open(public/case['image']) as im:im.verify()
        with np.load(public/case['data']) as arrays:
            assert arrays['candidate_coarse_signal_per_million'].shape==(2,128,128)
            assert np.isfinite(arrays['candidate_coarse_signal_per_million']).all()
    with zipfile.ZipFile(root/'blind_review/blind_review_packet.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())=={p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()}
    errors=[];screens=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport=dict(width=1200,height=1000),accept_downloads=True)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((public/'index.html').resolve().as_uri());page.wait_for_function('document.querySelector("#image").naturalWidth>0')
        assert page.locator('#case option').count()==len(manifest['cases'])
        for index in [0,len(manifest['cases'])//2,len(manifest['cases'])-1]:
            case=manifest['cases'][index];page.select_option('#case',case['case_id'])
            page.wait_for_function('(name)=>document.querySelector("#image").src.endsWith(name)&&document.querySelector("#image").complete&&document.querySelector("#image").naturalWidth>0',arg=case['image'])
            shot=out/f'offline_case_{index:03d}.png';page.screenshot(path=str(shot),full_page=True);screens.append(shot.as_posix())
        # Engineering data lives only in this ephemeral browser and is never imported as a review.
        page.fill('#reviewer','ENGINEERING TEST - NOT A HUMAN REVIEW');page.fill('#notes','Temporary browser export correctness check only')
        page.select_option('#assessment','insufficient_evidence');page.click('#save')
        assert page.evaluate('Object.keys(JSON.parse(localStorage.getItem("blindReviews"))).length')==1
        with page.expect_download() as download:page.click('#export')
        exported=json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
        assert len(exported)==1 and exported[0]['assessment']=='insufficient_evidence'
        page.evaluate('localStorage.clear()');browser.close()
    assert not errors,errors
    assert not (root/'blind_review/human_reviews.jsonl').exists()
    head=json.loads((root/'course_audit/initial_git_state.json').read_text(encoding='utf-8'))['head'];tracks=[]
    for directory in ['multitrack','multitrack_mutants']:
        index=pd.read_csv(root/'course_audit/committed_evidence/results'/directory/'index.csv')
        for i in [0,len(index)//2,len(index)-1]:
            row=index.iloc[i];name=row['path'];url=f'https://raw.githubusercontent.com/ysaaa-rain/dl-pro/{head}/{name}'
            response=requests.get(url,timeout=cfg['evidence']['network_timeout_seconds']);response.raise_for_status();payload=response.content
            assert len(payload)<=cfg['evidence']['metadata_max_bytes']
            oid=subprocess.check_output(['git','rev-parse',f'HEAD:{name}'],text=True).strip()
            assert hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()==oid
            local=out/'committed_track_samples'/directory/Path(name).name;local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(payload)
            with Image.open(local) as image:image.verify()
            tracks.append(dict(original_path=name,local_path=local.as_posix(),url=url,git_blob=oid,sha256=sha256(local),start=int(row.start),end=int(row.end)))
    write_json(out/'track_samples.json',tracks)
    write_json(out/'status.json',dict(cases_checked=len(manifest['cases']),packet_members_exact=True,png_and_npz_valid=True,edge_offline_ui_tested=True,browser_errors=errors,temporary_export_test_not_human_review=True,real_review_records_created=0,screenshots=screens,course_track_samples=len(tracks),track_selection='first, middle, last in each frozen genome index; six samples, not full visual validation'))
    print('All packet assets and offline Edge UI verified; six pinned track samples restored',flush=True)

if __name__=='__main__':main()
