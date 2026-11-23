# mini-ddpm-web

[打开网页](https://banligegit.github.io/mini-ddpm/) · [GitHub 仓库](https://github.com/banligegit/mini-ddpm)

## CIFAR-10 卷积 VAE

[题目五在线实验](https://banligegit.github.io/mini-ddpm/cifar-vae/)默认展示卡车原图、上轮 60 轮模型与本轮论文实验选定模型的 16×16×4 确定性重建。可切换至 8×8×4、查看首版与上轮，或在同一版本中对照两种尺寸。点击原图可移动三列同步的 12×12 像素放大框。先验页保留两份原始解码器的标准正态采样。各模型在浏览器内用 ONNX Runtime Web 的 WASM 后端推理。测试图按类别各取官方测试集首张，表中指标是 10,000 张测试图的均值。

从页面下载实验提交包。解压后在包含 `model.py` 的目录安装 `onnx`、`onnxruntime` 及提交包中的依赖，执行下列命令可更新本仓库的网页模型、测试图和指标；将路径参数替换为实际检出位置。重新导出前，数据目录应已有官方 CIFAR-10，或可联网首次下载。

```sh
python export_web.py --output /path/to/mini-ddpm/site/cifar-vae
```

从页面下载细节优化实验包，将其中的 `detail/` 放在上述训练项目根目录。导出优化版需先保留原版 JSON 和原图，再运行 `python detail/export_detail_web.py --output /path/to/mini-ddpm/site/cifar-vae`。导出脚本核验选定权重的 SHA-256，并将 ONNX 数值与 PyTorch 输出对照。

要导出本轮选定模型，从页面下载论文优化实验包并解压，在包的项目根目录运行 `python paper_detail/export_web_paper.py --output /path/to/mini-ddpm/site/cifar-vae`。网页仓库保留六个模型的比较资料。本轮导出脚本校验最终验证选择、权重摘要，以及十张示例在 ONNX 编码和解码后与 PyTorch 成图的最大绝对误差。

从本仓库根目录运行 `npx --yes http-server site -p 8080`，打开 `http://localhost:8080/cifar-vae/`。安装 Chrome、NumPy、Pillow 和 Python Playwright 后，`python cifar_vae/verify_browser.py http://localhost:8080/cifar-vae/` 可核对十张样例、同种子复现和移动布局。

## MNIST 手写数字生成器

随机类别或指定 0–9，选择 DDIM 10/20/50/200 步。种子为 0–4294967295 的整数。JS 用 32 位生成器和 Box–Muller 产生 784 个高斯值。Canvas 将 [-1,1] 映射到 [0,255]。每个模型在首次使用时加载并创建会话，随后复用。需要支持 WebGPU 的浏览器及 HTTPS 或 localhost。

## 本地运行

下载仓库后在根目录运行 `npx --yes http-server site -p 8080`，访问 http://localhost:8080 。site/ 已包含运行时、WASM 和两份 ONNX 模型，推理均在浏览器内完成。固定 ORT Web 1.23.2。JS、MJS 应使用 JavaScript MIME 类型，WASM 使用 application/wasm。

## 模型与权重导出

接口为固定 batch=1，输出 epsilon 和输入 x_t 均为 float32 [1,1,28,28]。t 为 int64 [1] 的原始时间索引，范围 0–999。条件模型额外接受 int64 [1] 的 label，值为 0–9。取前两题的 EMA 权重。源文件 model.py 和 diffusion.py 沿用训练工程。

使用 Python 3.11，安装 torch、numpy、onnx、onnxruntime，然后执行以下命令。checkpoint 必须包含 ema 与 config。原始 checkpoint 由题目一、二产生，部署直接使用导出的模型。

```sh
python export_models.py --unconditional PATH_TO_TASK1/last.pt --conditional PATH_TO_TASK2/conditional.pt
```

导出 opset 17。脚本在 PyTorch 与 Python ONNX Runtime CPU 上核对同一固定输入，并生成 site/fixtures/ 下共享的 little-endian float32 噪声文件、一步预测及四种步数最终张量、SHA-256 和实际索引。JS 采用完全相同的线性 beta 日程与截断重算噪声方向的 DDIM 更新，最后的干净端点累计信号系数为 1。

## 测试

安装本机 Chrome 和 Python playwright，执行 `python test_browser.py`。脚本临时启动本机静态服务，自动加载页面、测试两模式所有步数、比较 Canvas 量化后 784 个像素、验证种子重复性和 WebGPU 缺失提示。结果写入 browser-results.json。单步比较采用 t=999，条件标签 7。误差范围仅覆盖该固定测试张量。测试记录见 test-record.pdf。

## GitHub Pages

main 分支保留导出、测试和页面源文件。gh-pages 分支根目录发布 site/ 的内容。更新 site/ 后提交，再运行：

```sh
git push origin main
git subtree push --prefix site origin gh-pages
```

在仓库 Settings → Pages 选择 Deploy from a branch → gh-pages → / (root)。模型、脚本和 fixtures 均用相对 URL，支持 /mini-ddpm/ 子路径。运行时文件来自 npm 包 onnxruntime-web 的 dist 目录（ort.webgpu.min.mjs 和 ort-wasm-simd-threaded.asyncify.mjs/.wasm）。

参考 [GitHub Pages](https://docs.github.com/en/pages/quickstart)、[ONNX Runtime WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)、[DDIM](https://arxiv.org/abs/2010.02502)。
