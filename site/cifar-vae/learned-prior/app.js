import {draw,loadModels,noise,sample} from './engine.js';
const get = id => document.getElementById(id);
get('gen-form').addEventListener('submit', async event => {
  event.preventDefault();
  const controls = [...event.currentTarget.elements];
  const target = Number(get('target').value), seed = Number(get('seed').value);
  controls.forEach(el => { el.disabled=true; });
  get('progress').value=0; get('result').hidden=true; get('timing').textContent='';
  get('status').textContent='正在准备扩散模型和 VAE 解码器…';
  try {
    const models = await loadModels();
    const begin = performance.now();
    const {rgb} = await sample(models,noise(seed),target,async (i,n) => {
      get('progress').value = i;
      if (i===1 || i%5===0) {
        get('status').textContent='正在生成，第 ' + i + ' / ' + n + ' 步';
        await new Promise(resolve => setTimeout(resolve,0));
      }
    });
    draw(get('canvas'),rgb);
    get('result').hidden=false;
    get('caption').textContent=get('target').selectedOptions[0].textContent+' · 种子 '+seed;
    get('timing').textContent='首次模型加载 '+(models.loadMs/1000).toFixed(2)+' 秒 · 本次生成 '+((performance.now()-begin)/1000).toFixed(2)+' 秒';
    get('status').textContent='浏览器生成完成。';
  } catch (error) {
    get('status').textContent='生成失败。'+error.message;
  } finally {
    controls.forEach(el => { el.disabled=false; });
  }
});
