"""Compare the live browser reconstructions with the PyTorch examples and test seeded priors."""
import argparse
import base64
import io
import json
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def png_array(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="URL of the cifar-vae/ directory with trailing slash")
    args = parser.parse_args()
    url = args.url
    base = urllib.parse.urljoin(url, ".")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 980, "height": 920})
        page.set_default_timeout(90000)
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        response = page.goto(url, wait_until="domcontentloaded")
        assert response.ok, response.status
        page.locator("#recon-form button:enabled").wait_for()
        with urllib.request.urlopen(urllib.parse.urljoin(base, "results.json"), timeout=30) as stream:
            results = json.load(stream)
        maximum = {16: 0, 8: 0}
        for sample in results["samples"]:
            idx = sample["index"]
            page.select_option("#sample", str(idx))
            page.locator("#recon-form button").click()
            page.wait_for_function("document.querySelector('#status').textContent.includes('浏览器重建完成') || document.querySelector('#status').textContent.includes('本次推理未完成')")
            assert "浏览器重建完成" in page.locator("#status").inner_text(), page.locator("#status").inner_text()
            for side in (16, 8):
                actual = png_array(page.locator(f"#recon{side}").get_attribute("src"))
                path = urllib.parse.urljoin(base, f"samples/{idx}-recon{side}.png")
                with urllib.request.urlopen(path, timeout=30) as stream:
                    ref = np.asarray(Image.open(io.BytesIO(stream.read())).convert("RGB"), dtype=np.int16)
                maximum[side] = max(maximum[side], int(np.abs(ref - actual).max()))
        assert max(maximum.values()) <= 1, maximum
        page.locator("#choose-prior").click()
        page.locator("#prior8:visible").wait_for()
        def image_pair():
            return [page.locator(f"#prior{s}").evaluate("el => el.toDataURL()") for s in (16, 8)]
        first = image_pair()
        for seed in (2027, 2026):
            page.locator("#seed").fill(str(seed))
            page.locator("#prior-form button").click()
            page.wait_for_function("s => document.querySelector('#status').textContent.startsWith('种子 '+s+'，')", arg=seed)
            pair = image_pair()
            assert all((a == b) == (seed == 2026) for a, b in zip(first, pair)), seed
        page.set_viewport_size({"width": 360, "height": 750})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#choose-recon").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        print(json.dumps({"url": url, "browser": browser.version, "samples": len(results["samples"]),
                          "max_rgb_error_uint8": maximum, "seed_repeat": True, "mobile_layout": True}))
        browser.close()


if __name__ == "__main__":
    main()
