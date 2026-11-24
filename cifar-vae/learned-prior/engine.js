const here = path => new URL(path, import.meta.url);
let pending;

async function createModels() {
  const begin = performance.now();
  const response = await fetch(here('config.json'), {cache: 'no-cache'});
  if (!response.ok) throw Error('读取模型配置失败（HTTP ' + response.status + '）。');
  const config = await response.json();
  const ort = await import('../../vendor/ort.webgpu.min.mjs');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = here('../../vendor/').href;
  const options = {executionProviders: ['wasm']};
  const prior = await ort.InferenceSession.create(here('velocity.onnx?v=' + config.onnx_sha256.slice(0,12)).href, options);
  const decoder = await ort.InferenceSession.create(here(config.decoder_path + '?v=' + config.decoder_onnx_sha256.slice(0,12)).href, options);
  return {ort, prior, decoder, config, loadMs: performance.now()-begin};
}

export function loadModels() {
  if (!pending) pending = createModels().catch(error => { pending = undefined; throw error; });
  return pending;
}

export function noise(seed) {
  let state = seed >>> 0;
  function uniform() {
    state = (state + 0x6d2b79f5) >>> 0;
    let n = Math.imul(state ^ (state >>> 15), 1 | state);
    n ^= n + Math.imul(n ^ (n >>> 7), 61 | n);
    return (((n ^ (n >>> 14)) >>> 0) + .5) / 4294967296;
  }
  const values = new Float32Array(256);
  for (let i = 0; i < values.length; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(uniform())), angle = 2 * Math.PI * uniform();
    values[i] = radius * Math.cos(angle);
    values[i+1] = radius * Math.sin(angle);
  }
  return values;
}

export function randomClass(seed) {
  // A domain-separated stream leaves the image-noise stream and draw count unchanged.
  let state = (seed ^ 0x952616ad) >>> 0;
  let word;
  do {
    state = (state + 0x9e3779b9) >>> 0;
    word = Math.imul(state ^ (state >>> 16), 0x21f0aaad);
    word = Math.imul(word ^ (word >>> 15), 0x735a2d97);
    word = (word ^ (word >>> 15)) >>> 0;
  } while (word >= 4294967290);
  return word % 10;
}

export async function predict(models, x, t, label) {
  const {ort, prior} = models;
  const feeds = {
    x: new ort.Tensor('float32', x, [1,4,8,8]),
    t: new ort.Tensor('float32', Float32Array.of(t), [1]),
    label: new ort.Tensor('int64', BigInt64Array.of(BigInt(label)), [1]),
  };
  const {v} = await prior.run(feeds);
  const values = v.data.slice();
  v.dispose();
  Object.values(feeds).forEach(input => input.dispose());
  return values;
}

export async function sample(models, initial, target, onStep) {
  const {config, ort, decoder} = models;
  let x = initial.slice();
  if (x.length !== 256 || !Number.isInteger(target) || target < -1 || target > 9) throw Error('输入噪声或指定类别无效。');
  for (let i = 0; i < config.steps; i++) {
    const v = await predict(models, x, config.t[i], config.null_label);
    if (target >= 0) {
      const conditional = await predict(models, x, config.t[i], target);
      for (let j=0; j<v.length; j++) v[j] += config.guidance * (conditional[j]-v[j]);
    }
    const next = new Float32Array(x.length);
    for (let j=0; j<x.length; j++) {
      const clean = config.alpha[i]*x[j] - config.sigma[i]*v[j];
      const epsilon = config.sigma[i]*x[j] + config.alpha[i]*v[j];
      next[j] = config.alpha[i+1]*clean + config.sigma[i+1]*epsilon;
    }
    x = next;
    if (onStep) await onStep(i+1, config.steps);
  }
  const latent = x.map((value,j) => value*config.std[Math.floor(j/64)] + config.mean[Math.floor(j/64)]);
  const z = new ort.Tensor('float32', latent, [1,4,8,8]);
  const out = await decoder.run({z});
  const rgb = out.rgb.data.slice();
  z.dispose();
  out.rgb.dispose();
  return {latent, rgb};
}

export function draw(canvas, rgb) {
  const ctx = canvas.getContext('2d');
  const frame = ctx.createImageData(32,32);
  for (let j=0; j<1024; j++) {
    for (let c=0; c<3; c++) frame.data[4*j+c] = Math.round(rgb[c*1024+j]*255);
    frame.data[4*j+3] = 255;
  }
  ctx.putImageData(frame,0,0);
}
