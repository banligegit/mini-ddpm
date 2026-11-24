"""Verify the deployed CIFAR-10 originals, two selected ONNX pairs, previews, crops and seeded priors."""
import argparse
import base64
import io
import json
from pathlib import Path
import urllib.parse

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def png_array(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="URL of the cifar-vae/ directory with trailing slash")
    parser.add_argument("--expected-catalog", type=Path, help="Compare public measurements with the selected local release")
    args = parser.parse_args()
    base = urllib.parse.urljoin(args.url, ".")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 980, "height": 920})
        page.set_default_timeout(90000)
        errors, models_loaded = [], set()
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("response", lambda r: models_loaded.add(urllib.parse.urlsplit(r.url).path.split("/")[-1])
                if ".onnx" in r.url and r.ok else None)
        assert page.goto(args.url, wait_until="domcontentloaded").ok
        page.locator("#recon-form button:enabled").wait_for()
        catalog = page.evaluate("""async () => {
          const response = await fetch('./best-results.json', {cache:'no-cache'});
          if (!response.ok) throw Error('Metrics returned HTTP ' + response.status);
          return response.json();
        }""")
        assert catalog["test_image_count"] == 10000 and len(catalog["samples"]) == 10
        assert set(catalog["models"]) == {"16", "8"}
        if args.expected_catalog:
            assert catalog == json.loads(args.expected_catalog.read_text(encoding="utf-8")), "Wrong deployed weights/metrics"
        assert page.locator("#sample").input_value() == "11"
        assert page.locator("#recon-panel select").count() == 1
        assert page.locator("#prior-panel select").count() == 0
        assert page.locator("#metrics tr").count() == 2
        downloads = page.locator("#report-heading + ul a").evaluate_all("links => links.map(a=>a.getAttribute('href'))")
        assert downloads == ["./report.pdf", "./submission.zip"], downloads
        maximum = {16: 0, 8: 0}
        for row, side in enumerate((16, 8)):
            m = catalog["models"][str(side)]
            cells = page.locator("#metrics tr").nth(row).locator("td").all_inner_texts()
            assert cells == [f"{side} × {side} × 4", f"{m['parameters']:,}", f"{m['psnr']:.2f}",
                             f"{m['ssim']:.4f}", f"{m['local_detail_mae']:.2f}"], cells
        for sample in catalog["samples"]:
            idx = sample["index"]
            page.select_option("#sample", str(idx))
            original = page.locator("#original")
            original.evaluate("el=>el.decode()")
            assert f"/{idx}-original.png" in original.get_attribute("src")
            references = {}
            for side in (16, 8):
                image = page.locator(f"#recon{side}")
                assert f"{idx}-best{side}.png" in image.get_attribute("src")
                preview = image.evaluate("""async el => {
                  await el.decode();
                  const canvas = document.createElement('canvas');
                  canvas.width = el.naturalWidth;
                  canvas.height = el.naturalHeight;
                  canvas.getContext('2d').drawImage(el, 0, 0);
                  return canvas.toDataURL();
                }""")
                references[side] = png_array(preview)
            page.locator("#recon-form button").click()
            page.wait_for_function("['浏览器重建完成','本次推理未完成'].some(s=>document.querySelector('#status').textContent.includes(s))")
            assert "浏览器重建完成" in page.locator("#status").inner_text(), page.locator("#status").inner_text()
            xy = page.locator("#crop-coordinates").inner_text().split("(")[1].rstrip(")")
            x, y = map(int, xy.split(","))
            for side in (16, 8):
                actual = png_array(page.locator(f"#recon{side}").get_attribute("src"))
                maximum[side] = max(maximum[side], int(np.abs(actual - references[side]).max()))
                cropped = png_array(page.locator(f"#zoom-recon{side}").evaluate("el=>el.toDataURL()"))
                assert np.array_equal(cropped, actual[y:y+12, x:x+12]), (idx, side)
        assert max(maximum.values()) <= 1, maximum
        assert models_loaded == {f"best-{part}{side}.onnx" for part in ("encoder", "decoder") for side in (16, 8)}
        crop = page.locator("#crop-source")
        crop.click(position={"x": 1, "y": 1})
        assert "(0, 0)" in page.locator("#crop-coordinates").inner_text()
        crop.press("ArrowRight")
        assert "(1, 0)" in page.locator("#crop-coordinates").inner_text()
        page.locator("#choose-prior").click()
        for img in ("#prior-real", "#fixed-prior16", "#fixed-prior8"):
            page.locator(img).evaluate("el=>el.decode()")
            assert page.locator(img).is_visible()
        first = None
        for seed in (2026, 2027, 2026):
            page.locator("#seed").fill(str(seed))
            page.locator("#prior-form button").click()
            page.wait_for_function("s=>document.querySelector('#status').textContent.startsWith('种子 '+s+'，')", arg=seed)
            pair = [png_array(page.locator(f"#prior{s}").evaluate("el=>el.toDataURL()")) for s in (16, 8)]
            assert all(grid.shape == (128, 256, 3) for grid in pair)
            if first is None:
                first = pair
            else:
                assert all(np.array_equal(a, b) == (seed == 2026) for a, b in zip(first, pair))
        assert page.locator("#prior-real").is_visible()
        page.set_viewport_size({"width": 360, "height": 750})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#choose-recon").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.goto("about:blank")
        page.goto(base + "#priors", wait_until="domcontentloaded")
        page.locator("#prior-panel:visible").wait_for()
        assert page.locator("#fixed-prior8").is_visible()
        assert not errors, errors
        print(json.dumps({"url": args.url, "browser": browser.version, "samples": len(catalog["samples"]),
                          "checkpoints": {side: m["checkpoint_sha256"] for side, m in catalog["models"].items()},
                          "models_loaded": sorted(models_loaded), "max_rgb_error_uint8": maximum,
                          "downloads": downloads, "synchronized_crops": True, "seed_repeat": True,
                          "fixed_real_reference": True, "mobile_layout": True}))
        browser.close()


if __name__ == "__main__":
    main()
