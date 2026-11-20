const $=q=>document.querySelector(q);
const form=$('form'),status=$('#status'),button=$('button');
for(let i=0;i<10;i++)$('#digit').add(new Option(i,i,false,i===7));
$('#mode').onchange=()=>$('#digit').disabled=$('#mode').value==='unconditional';
try{
  if(!isSecureContext)throw Error('请通过 HTTPS 或 localhost 打开网页。');
  if(!navigator.gpu)throw Error('当前浏览器没有 WebGPU 接口。请使用 Chrome 或 Edge，检查硬件加速设置。');
  if(!await navigator.gpu.requestAdapter())throw Error('找不到可用 GPU。请检查驱动及浏览器硬件加速设置。');
  const {load,noise,sample,indices,draw}=await import('./engine.js');
  button.disabled=false;status.textContent='WebGPU 就绪';
  form.onsubmit=async event=>{
    event.preventDefault();
    const mode=$('#mode').value,k=Number($('#steps').value),digit=Number($('#digit').value),seed=Number($('#seed').value);
    for(const el of form.elements)if('disabled' in el)el.disabled=true;
    $('#timing').textContent='';$('progress').value=0;status.textContent='加载模型…';
    try{
      const {session,loadMs}=await load(mode);
      const begin=performance.now();
      const x=await sample(session,noise(seed),digit,indices(k),(i,n)=>{status.textContent='采样 '+i+' / '+n;$('progress').value=i/n;});
      draw($('canvas'),x);status.textContent='生成完成';
      $('#timing').textContent='首次模型加载 '+(loadMs/1000).toFixed(2)+' 秒 · 本次生成 '+((performance.now()-begin)/1000).toFixed(2)+' 秒';
    }catch(err){status.textContent='生成失败：'+err.message;}
    finally{for(const el of form.elements)if('disabled' in el)el.disabled=false;$('#mode').onchange();}
  };
}catch(err){status.textContent=err.message;}
