"""Run the real one-image UI and compare browser float tensors with the shared FP32 PyTorch fixtures."""
import argparse
import json
import urllib.parse
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url",help="URL of the learned-prior/ page, ending in /")
    args=parser.parse_args()
    with sync_playwright() as pw:
        browser=pw.chromium.launch(channel="chrome",headless=True)
        page=browser.new_page(viewport={"width":1000,"height":850})
        page.set_default_timeout(120000)
        errors=[]
        page.on("pageerror",lambda e: errors.append(str(e)))
        response=page.goto(args.url,wait_until="domcontentloaded")
        assert response.ok
        assert page.locator("#target option").count()==12
        for image in page.locator(".showcase img").all(): image.evaluate("el=>el.decode()")
        result=page.evaluate("""async () => {
          const {loadModels,predict,sample,draw}=await import('./engine.js?v=class-draw');
          const models=await loadModels(), config=models.config;
          async function fixture(name) {
            const r=await fetch('./fixture/'+name+'.f32');
            if(!r.ok) throw Error(name+' '+r.status);
            return new Float32Array(await r.arrayBuffer());
          }
          function error(a,b) {
            if (a.length!==b.length || !a.every(Number.isFinite)) throw Error('invalid tensor');
            const d=a.map((v,i)=>Math.abs(v-b[i]));
            return {mean:d.reduce((s,v)=>s+v,0)/d.length,max:Math.max(...d)};
          }
          const initial=await fixture('noise');
          const result={steps:config.steps,guidance:config.guidance,model:config.checkpoint_sha256,provider:'wasm'};
          for(const [mode,target] of [['unconditional',-1],['conditional',config.fixture_class]]) {
            const single=await predict(models,initial,config.t[0],target<0?config.null_label:target);
            const out=await sample(models,initial,target);
            const expected=await fixture(mode+'-rgb');
            const canvases=[out.rgb,expected].map(rgb=>{
              const c=document.createElement('canvas'); c.width=c.height=32; draw(c,rgb);
              return c.getContext('2d').getImageData(0,0,32,32).data;
            });
            result[mode]={v:error(single,await fixture(mode+'-v')),
              z:error(out.latent,await fixture(mode+'-z')),rgb:error(out.rgb,expected),
              max_byte:Math.max(...Array.from(canvases[0],(v,i)=>Math.abs(v-canvases[1][i])))};
          }
          return result;
        }""")
        assert result["steps"]==60 and result["guidance"]==6
        for mode in ("unconditional","conditional"):
            assert max(result[mode][k]["max"] for k in ("v","z","rgb"))<3e-4, result
            assert result[mode]["max_byte"]<=1, result
        frames=[]
        picked=page.evaluate("async()=>{const m=await import('./engine.js?v=class-draw');return m.randomClass(2026)}")
        for target,seed in ((-1,2026),(7,2026),(7,2026),(7,2027),("draw",2026),(picked,2026)):
            page.select_option("#target",str(target))
            page.fill("#seed",str(seed))
            page.locator("#gen-form button").click()
            page.wait_for_function("['浏览器生成完成。','生成失败。'].some(s=>document.getElementById('status').textContent.startsWith(s))")
            assert "完成" in page.locator("#status").inner_text(),page.locator("#status").inner_text()
            assert page.locator("#result").is_visible() and page.locator("#progress").get_attribute("value")=="60"
            frames.append(page.locator("#canvas").evaluate("c=>c.toDataURL()"))
            if target=="draw": assert "抽中 "+str(picked)+" ·" in page.locator("#caption").inner_text()
        assert frames[0]!=frames[1]==frames[2]!=frames[3]
        assert frames[4]==frames[5]
        result["last_ui_timing"]=page.locator("#timing").inner_text()
        page.set_viewport_size({"width":360,"height":800})
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        root=urllib.parse.urljoin(args.url,"../")
        page.goto(root,wait_until="domcontentloaded")
        assert page.locator("a[href='./learned-prior/']").is_visible()
        page.locator("#recon-form button:enabled").wait_for()
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.locator("#choose-prior").click()
        page.locator("#fixed-prior8").evaluate("img=>img.decode()")
        page.locator("#fixed-prior8").wait_for(state="visible")
        assert not errors, errors
        result.update(browser=browser.version,url=args.url,seed_repeat=True,seed_change=True,condition_change=True,seeded_class_draw=picked)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        browser.close()


if __name__=="__main__": main()
