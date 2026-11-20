import * as ort from './vendor/ort.webgpu.min.mjs';
ort.env.wasm.numThreads=1;
ort.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
const schedule=new Float32Array(1000);
let alpha=1;
for(let t=0;t<1000;t++){
  alpha*=1-(0.0001+(0.02-0.0001)*t/999);
  schedule[t]=alpha;
}
export const indices=k=>Array.from({length:k},(_,i)=>Math.round((k-1-i)*999/(k-1)));
export function noise(seed){
  let state=seed>>>0;
  function uniform(){
    state=(state+0x6D2B79F5)>>>0;
    let n=Math.imul(state^(state>>>15),1|state);
    n^=n+Math.imul(n^(n>>>7),61|n);
    return (((n^(n>>>14))>>>0)+0.5)/4294967296;
  }
  const x=new Float32Array(784);
  for(let i=0;i<x.length;i+=2){
    const r=Math.sqrt(-2*Math.log(uniform())),theta=2*Math.PI*uniform();
    x[i]=r*Math.cos(theta);x[i+1]=r*Math.sin(theta);
  }
  return x;
}
const sessions=new Map();
export async function load(mode){
  if(sessions.has(mode))return sessions.get(mode);
  if(!isSecureContext)throw Error('WebGPU 需要 HTTPS 或 localhost。');
  if(!navigator.gpu)throw Error('此浏览器没有 WebGPU 接口。请使用支持 WebGPU 的 Chrome 或 Edge，并检查硬件加速设置。');
  if(!await navigator.gpu.requestAdapter())throw Error('没有可用的 WebGPU 适配器。请检查 GPU 驱动和浏览器硬件加速设置。');
  const begin=performance.now();
  const session=await ort.InferenceSession.create(new URL('./models/'+mode+'.onnx',import.meta.url).href,{executionProviders:['webgpu']});
  const entry={session,loadMs:performance.now()-begin};sessions.set(mode,entry);return entry;
}
export async function predict(session,x,t,label){
  const int=n=>new ort.Tensor('int64',BigInt64Array.of(BigInt(n)),[1]);
  const feeds={x_t:new ort.Tensor('float32',x,[1,1,28,28]),t:int(t)};
  if(session.inputNames.includes('label'))feeds.label=int(label);
  const {epsilon}=await session.run(feeds);
  return epsilon.data;
}
export async function sample(session,start,label,seq,update=()=>{}){
  let x=start.slice();
  for(const [i,t] of seq.entries()){
    const e=await predict(session,x,t,label),s=seq[i+1];
    const at=schedule[t], ap=s===undefined?1:schedule[s];
    const u=Math.sqrt(at),v=Math.sqrt(1-at),a=Math.sqrt(ap),b=Math.sqrt(1-ap);
    for(let j=0;j<x.length;j++){
      const clean=Math.max(-1,Math.min(1,(x[j]-v*e[j])/u));
      const corrected=(x[j]-u*clean)/v;
      x[j]=a*clean+b*corrected;
    }
    update(i+1,seq.length);
  }
  return x;
}
export function draw(canvas,x){
  const ctx=canvas.getContext('2d'), pixels=ctx.createImageData(28,28);
  for(let i=0;i<784;i++){const gray=Math.round((x[i]+1)*127.5);pixels.data.set([gray,gray,gray,255],i*4);}
  ctx.putImageData(pixels,0,0);
}
