"""Check the live HTTPS site in a new Chrome profile and save actual Canvas output."""
import base64
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL='https://banligegit.github.io/mini-ddpm/'
with sync_playwright() as p:
    b=p.chromium.launch(channel='chrome',headless=False,args=['--window-position=-2000,0'])
    page=b.new_page()
    page.on('console',lambda m:print(m.text[:400],flush=True) if m.type=='error' else None)
    for attempt in range(8):
        resp=page.goto(URL+'?v=3',timeout=45000)
        if resp.ok and page.locator('#mode').count():break
        time.sleep(8)
    assert resp.ok
    measurements={}
    for mode in ['conditional','unconditional']:
        page.select_option('#mode',mode)
        page.select_option('#steps','20')
        page.locator('button').click(timeout=60000)
        page.wait_for_function("['生成完成','生成失败'].some(s=>document.querySelector('#status').textContent.startsWith(s))",timeout=240000)
        assert page.locator('#status').inner_text()=='生成完成',page.locator('#status').inner_text()
        measurements[mode]=page.locator('#timing').inner_text()
        src=page.locator('canvas').evaluate("el=>el.toDataURL('image/png').split(',')[1]")
        Path('live-'+mode+'.png').write_bytes(base64.b64decode(src))
    Path('live-results.json').write_text(json.dumps(dict(url=URL,status=resp.status,browser=b.version,measurements=measurements),ensure_ascii=False,indent=2),encoding='utf8')
    b.close()
