# 浏览器中的 CIFAR-10 学习先验

[打开拓展网页](https://banligegit.github.io/mini-ddpm/cifar-vae/learned-prior/)。固定网格取自最终 30,000 次更新 EMA 实验的 seed 2026，按原顺序各取完整 32 张。真实参考图为官方测试集前 32 张。固定生成图使用实验端的 PyTorch/CUDA。网页的抽样实现使用 JavaScript 与 ONNX Runtime Web 的 WASM 后端。

扩散网络使用 velocity.onnx，配套的 8 × 8 解码器保存在同目录 frozen-decoder8.onnx。其源权重与潜变量标准化参数通过 SHA-256 固定到扩散训练时使用的版本。输入 x 为 float32 [1,4,8,8]，t 为 float32 [1]，label 为 int64 [1]。输出 v 为 float32 [1,4,8,8]。标签 0–9 依次为飞机、汽车、鸟、猫、鹿、狗、蛙、马、船、卡车。标签 10 调用同一网络的空条件预测。每步由 v 得到 clean = αx − σv 和 eps = σx + αv，再按下个时间点合成新的潜变量。指定类别时合并两次预测 v_null + s(v_label − v_null)，验证集选中的 s 为 6。60 步采用连续余弦 α、σ 和确定性 DDIM。终点按冻结训练编码的通道均值和标准差逆标准化，送入已有的 8 × 8 VAE 解码器。浮点时间和系数保存在 config.json。

“随机抽类别（引导生成）”用独立于图像噪声的确定性随机流均匀抽出 0–9 中一类，再执行类别引导。画布下显示抽中类别。同种子改成手动指定该类可生成相同像素。“随机（无条件）”全程使用空标签。

在公开网页验证已发布模型需要 Python、Playwright 和本机 Chrome。下面的命令覆盖两种模式的速度预测、完整潜变量、最终 RGB 及 8 位 Canvas，另测网页种子重复、换种子和换类。跨语言测试共读 fixture/noise.f32，固定网格不参与数值对齐。

```sh
python -m pip install playwright
python cifar_vae/learned_prior/verify_browser.py https://banligegit.github.io/mini-ddpm/cifar-vae/learned-prior/
```

重新导出需本地题五训练目录，其中包含 prior_generation/diffusion/extension/checkpoints/ema-30000.pt、最终 guidance-validation.json、extension/final 的图像 NPZ、冻结 VAE 及校准记录。脚本核验两份 PyTorch 权重的 SHA-256，以 FP32 PyTorch 另算共享输入的参考端点，再在 Python 中逐项核验原模型与 ONNX。安装 PyTorch、NumPy、Pillow、onnx、onnxruntime 后，在当前仓库根目录运行以下命令。

```sh
python cifar_vae/learned_prior/export_onnx.py --task5 /path/to/task5 --site site/cifar-vae
npx --yes http-server site -p 8080
python cifar_vae/learned_prior/verify_browser.py http://localhost:8080/cifar-vae/learned-prior/
```
