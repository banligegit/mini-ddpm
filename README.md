# mini-ddpm-web

[打开网页](https://banligegit.github.io/mini-ddpm/) · [GitHub 仓库](https://github.com/banligegit/mini-ddpm)

## CIFAR-10 卷积 VAE

[原图与重建](https://banligegit.github.io/mini-ddpm/cifar-vae/#comparison)在同一行展示真实照片和两种潜变量尺寸的确定性重建。每种尺寸使用验证集选定的一份权重。十个样例各取一类在官方测试集中的首张照片。点击原图可移动三列同步的 12×12 像素放大框。指标统计涵盖官方测试集全部 10,000 张图。

[先验生成](https://banligegit.github.io/mini-ddpm/cifar-vae/#priors)并排展示测试集前 32 张照片和两个解码器各自的连续 32 个样本。每组生成图都来自标准正态潜变量，与参考照片没有逐张对应关系。默认图片取自正式评估。改变网页种子可用 ONNX Runtime Web 的 WASM 后端在浏览器另行抽样。

[学习先验（拓展）](https://banligegit.github.io/mini-ddpm/cifar-vae/learned-prior/)用同一份 30,000 步 EMA 潜空间扩散权重生成无条件或指定 CIFAR 类别的图片，60 次确定性 DDIM 更新后使用扩散训练时固定的 8 × 8 VAE 解码器。页面展示两组未筛选固定样本，并可在浏览器实际运行单图生成。[导出与浏览器对齐说明](cifar_vae/learned_prior/README.md)

[实验报告](https://banligegit.github.io/mini-ddpm/cifar-vae/report.pdf)与[提交包](https://banligegit.github.io/mini-ddpm/cifar-vae/submission.zip)提供选型记录、训练与评估代码及两份权重。在提交包的项目根目录安装其中的依赖，以及 `onnx`、`onnxruntime`，按包内说明准备 CIFAR-10，然后运行以下命令。将最后一个参数替换成本仓库的实际路径。

```sh
python reconstruction_next/export_web.py --output /path/to/mini-ddpm/site/cifar-vae
```

导出器检查验证集选定权重及官方测试记录的 SHA-256，生成四个 ONNX 文件、模型指标、十组原图与重建图、三个先验页网格，并核验 ONNX 与 PyTorch 的编码、解码及整条重建路径。在本仓库根目录运行 `npx --yes http-server site -p 8080` 后打开 `http://localhost:8080/cifar-vae/`。安装 Chrome、NumPy、Pillow 和 Python Playwright 后，运行 `python cifar_vae/verify_browser.py http://localhost:8080/cifar-vae/` 可核对四个模型、十组重建、固定种子、网页入口与移动布局。

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
