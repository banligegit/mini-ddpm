import {load,predict,indices,sample,noise,draw} from './engine.js';
async function fixed(name){
  const r=await fetch('./fixtures/'+name+'.f32');
  if(!r.ok)throw Error('fixture '+r.status);
  return new Float32Array(await r.arrayBuffer());
}
function errors(a,b){
  if(a.length!==b.length)throw Error('shape mismatch');
  let sum=0,max=0;
  a.forEach((v,i)=>{const delta=Math.abs(v-b[i]);if(!Number.isFinite(delta))throw Error('nonfinite');sum+=delta;max=Math.max(max,delta);});
  return {mean:sum/a.length,max};
}
function grayscale(x){
  const canvas=document.createElement('canvas');canvas.width=canvas.height=28;draw(canvas,x);
  return canvas.getContext('2d').getImageData(0,0,28,28).data.filter((_,i)=>i%4===0);
}
export async function parity(){
  const start=await fixed('initial'),meta=await (await fetch('./fixtures/manifest.json')).json();
  const adapter=await navigator.gpu.requestAdapter();
  const result={browser:navigator.userAgent,adapter:adapter.info?.description,vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,modes:{}};
  for(const mode of ['unconditional','conditional']){
    const {session,loadMs}=await load(mode);
    const eps=await predict(session,start,999,7);
    const run={loadMs,eps:errors(eps,await fixed(mode+'-eps')),steps:{}};result.modes[mode]=run;
    for(const k of [10,20,50,200]){
      const seq=indices(k);
      if(JSON.stringify(seq)!==JSON.stringify(meta.steps[k]))throw Error('time indices differ');
      const begin=performance.now(),x=await sample(session,start,7,seq);
      const ms=performance.now()-begin,ref=await fixed(mode+'-'+k);
      const img=grayscale(x),gold=grayscale(ref);
      run.steps[k]={ms,tensor:errors(x,ref),pixel:errors(img,gold),changedPixels:img.reduce((n,v,i)=>n+Number(v!==gold[i]),0)};
    }
  }
  const {session}=await load('conditional');
  const seeded=async seed=>sample(session,noise(seed),7,indices(10));
  const a=await seeded(2026);
  result.sameSeed=errors(a,await seeded(2026));result.otherSeed=errors(a,await seeded(2027));
  return result;
}
