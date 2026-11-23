const get = id => document.getElementById(id);
const sides = [16, 8];
const sessions = new Map();
let catalog, runtime, busy = false, priorShown = false;

async function loadRuntime() {
  if (!runtime) {
    runtime = await import('../vendor/ort.webgpu.min.mjs');
    runtime.env.wasm.numThreads = 1;
    runtime.env.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
  }
  return runtime;
}

async function session(side, part) {
  const key = part + side;
  if (!sessions.has(key)) {
    const ort = await loadRuntime();
    const url = new URL('./models/' + key + '.onnx', import.meta.url).href;
    sessions.set(key, await ort.InferenceSession.create(url, {executionProviders: ['wasm']}));
  }
  return sessions.get(key);
}

function makeNormal(seed) {
  let state = seed >>> 0;
  function uniform() {
    state = (state + 0x6d2b79f5) >>> 0;
    let n = Math.imul(state ^ (state >>> 15), 1 | state);
    n ^= n + Math.imul(n ^ (n >>> 7), 61 | n);
    return (((n ^ (n >>> 14)) >>> 0) + 0.5) / 4294967296;
  }
  return size => {
    const values = new Float32Array(size);
    for (let i = 0; i < size; i += 2) {
      const radius = Math.sqrt(-2 * Math.log(uniform()));
      const angle = 2 * Math.PI * uniform();
      values[i] = radius * Math.cos(angle);
      values[i + 1] = radius * Math.sin(angle);
    }
    return values;
  };
}

function rgbImage(chw) {
  const result = new ImageData(32, 32);
  for (let i = 0; i < 1024; i++) {
    for (let c = 0; c < 3; c++) result.data[4*i+c] = Math.round(chw[c*1024+i]*255);
    result.data[4*i+3] = 255;
  }
  return result;
}

function chosen() { return catalog.samples[get('sample').selectedIndex]; }
function showExample() {
  const sample = chosen();
  for (const name of ['original', 'recon16', 'recon8']) {
    get(name).src = './samples/' + sample.index + '-' + name + '.png';
  }
  get('status').textContent = 'CIFAR-10 测试集索引 ' + sample.index + '。图中展示两份训练权重在 Python 中的确定性重建。';
}
function setBusy(value) {
  busy = value;
  for (const row of document.querySelectorAll('button,select,input')) row.disabled = value;
}
async function run(action) {
  if (busy) return;
  setBusy(true);
  try {
    await action();
  } catch (error) {
    get('status').textContent = '本次推理未完成。' + error.message;
  } finally {
    setBusy(false);
  }
}

async function reconstruct() {
  get('status').textContent = '正在读取图像并加载编码器和解码器…';
  const original = get('original');
  await original.decode();
  const work = document.createElement('canvas');
  work.width = work.height = 32;
  const ctx = work.getContext('2d', {willReadFrequently:true});
  ctx.drawImage(original, 0, 0);
  const rgb = ctx.getImageData(0, 0, 32, 32).data;
  const numbers = new Float32Array(3072);
  for (let i = 0; i < 1024; i++) {
    for (let c = 0; c < 3; c++) numbers[c*1024+i] = rgb[4*i+c]/255;
  }
  const ort = await loadRuntime();
  const x = new ort.Tensor('float32', numbers, [1,3,32,32]);
  const previews = [];
  for (const side of sides) {
    const encoder = await session(side, 'encoder');
    const decoder = await session(side, 'decoder');
    const {mu} = await encoder.run({x});
    const {rgb} = await decoder.run({z:mu});
    ctx.putImageData(rgbImage(rgb.data), 0, 0);
    previews.push(work.toDataURL('image/png'));
  }
  sides.forEach((side,i) => { get('recon' + side).src = previews[i]; });
  get('status').textContent = '浏览器重建完成。两份模型分别编码同一张原图，再解码各自的后验均值。';
}

async function generate() {
  const seed = Number(get('seed').value);
  const ort = await loadRuntime();
  const grids = [];
  for (const side of sides) {
    get('status').textContent = '正在用 ' + side + ' × ' + side + ' 解码器生成 16 张图…';
    const decoder = await session(side, 'decoder');
    const normal = makeNormal(seed);
    const grid = document.createElement('canvas');
    grid.width = grid.height = 128;
    const ctx = grid.getContext('2d');
    for (let i = 0; i < 16; i++) {
      const z = new ort.Tensor('float32', normal(4*side*side), [1,4,side,side]);
      const {rgb} = await decoder.run({z});
      ctx.putImageData(rgbImage(rgb.data), (i%4)*32, Math.floor(i/4)*32);
    }
    grids.push(grid);
  }
  sides.forEach((side, i) => {
    const canvas = get('prior' + side);
    canvas.getContext('2d').drawImage(grids[i], 0, 0);
    canvas.hidden = false;
    canvas.setAttribute('aria-label', side + ' × ' + side + ' 解码器生成的 16 张未筛选图像');
  });
  priorShown = true;
  get('status').textContent = '种子 ' + seed + '，每组展示该尺寸下连续抽取的前 16 张图像。';
}

function switchMode(prior) {
  get('prior-panel').hidden = !prior;
  get('recon-panel').hidden = prior;
  get('choose-prior').setAttribute('aria-pressed', prior);
  get('choose-recon').setAttribute('aria-pressed', !prior);
  if (prior && !priorShown) {
    run(generate);
  } else if (prior) {
    get('status').textContent = '改变种子后点击生成，可观察另一组未筛选的先验样本。';
  } else {
    showExample();
  }
}

async function init() {
  const res = await fetch('./results.json');
  if (!res.ok) throw Error('无法读取测试结果（HTTP ' + res.status + '）。');
  catalog = await res.json();
  for (const s of catalog.samples) get('sample').add(new Option(s.name + ' · #' + s.index, s.index));
  for (const side of sides) {
    const m = catalog.models[side], tr = get('metrics').insertRow();
    for (const text of [side + ' × ' + side + ' × 4', (4*side*side).toLocaleString('en-US'), m.psnr.toFixed(2), m.ssim.toFixed(4)]) {
      tr.insertCell().textContent = text;
    }
  }
  get('sample').addEventListener('change', showExample);
  get('recon-form').addEventListener('submit', e => { e.preventDefault(); run(reconstruct); });
  get('prior-form').addEventListener('submit', e => { e.preventDefault(); run(generate); });
  get('choose-recon').addEventListener('click', () => switchMode(false));
  get('choose-prior').addEventListener('click', () => switchMode(true));
  showExample();
  setBusy(false);
}
init().catch(error => { get('status').textContent = error.message; });
