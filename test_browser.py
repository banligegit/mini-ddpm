"""Start a local static server and exercise the published UI and WebGPU implementation."""
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    SimpleHTTPRequestHandler.extensions_map.update({".js":"text/javascript",".mjs":"text/javascript",".wasm":"application/wasm"})
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory='site'))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f'http://127.0.0.1:{server.server_port}/'
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='chrome',headless=False,args=['--window-position=-2000,0','--no-first-run'])
        page=browser.new_page()
        page.on('console',lambda m: print('browser:',m.text[:500],flush=True) if m.type in ['error','warning'] else None)
        page.goto(base)
        result=page.evaluate("async()=> (await import('./parity.js')).parity()")
        result['browser_version']=browser.version
        result['page_origin']='localhost'
        page.select_option('#steps','10')
        page.locator('button').click()
        page.wait_for_function("document.querySelector('#status').textContent==='生成完成'",timeout=60000)
        result['ui']=page.locator('#timing').inner_text()
        result['webgpu']=True
        fallback=browser.new_page()
        fallback.add_init_script("Object.defineProperty(navigator,'gpu',{value:undefined})")
        fallback.goto(base)
        fallback.wait_for_function("document.querySelector('#status').textContent.includes('没有 WebGPU')")
        result['unsupported']=fallback.locator('#status').inner_text()
        Path('browser-results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
        browser.close()
    server.shutdown()
if __name__=='__main__':main()

