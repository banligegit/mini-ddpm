"""Export EMA weights and a shared, little-endian float32 parity fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from model import TinyUNet
from diffusion import Diffusion

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--unconditional',default='../mini-ddpm/runs/main/last.pt')
    p.add_argument('--conditional',default='../mini-ddpm-conditional/weights/conditional.pt')
    a=p.parse_args()
    root=Path('site'); models=root/'models'; fixtures=root/'fixtures'
    for folder in (models,fixtures): folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    torch.randn(1,1,28,28).numpy().astype('<f4').tofile(fixtures/'initial.f32')
    start=torch.from_numpy(np.fromfile(fixtures/'initial.f32','<f4').reshape(1,1,28,28))
    d=Diffusion()
    meta={}
    for mode,path in vars(a).items():
        ck=torch.load(path,map_location='cpu',weights_only=True)
        net=TinyUNet(base=ck['config']['base'],num_classes=10 if mode=='conditional' else None).eval()
        net.load_state_dict(ck['ema'])
        labels=(torch.tensor([7]),) if mode=='conditional' else ()
        args=(start,torch.tensor([999]),*labels)
        names=['x_t','t']+(['label'] if labels else [])
        target=models/(mode+'.onnx')
        torch.onnx.export(net,args,str(target),dynamo=False,opset_version=17,input_names=names,output_names=['epsilon'])
        sess=ort.InferenceSession(str(target),providers=['CPUExecutionProvider'])
        def desc(io): return {p.name:[p.type,p.shape] for p in io}
        with torch.inference_mode():
            ref=net(*args).numpy()
            pred=sess.run(None,{key:v.numpy() for key,v in zip(names,args)})[0]
            err=np.abs(pred-ref)
            np.testing.assert_allclose(pred,ref,atol=2e-4,rtol=2e-4)
            ref.astype('<f4').tofile(fixtures/(mode+'-eps.f32'))
            for count in [10,20,50,200]:
                seq=d.sampling_indices(count)[::-1]
                x=start
                for t,s in zip(seq,seq[1:]+[-1]):
                    x=d.ddim_step(x,net(x,torch.tensor([t]),*labels),t,s)
                x.numpy().astype('<f4').tofile(fixtures/f'{mode}-{count}.f32')
        meta[mode]=dict(inputs=desc(sess.get_inputs()),outputs=desc(sess.get_outputs()),mean=float(err.mean()),max=float(err.max()),onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    meta['steps']={k:d.sampling_indices(k)[::-1] for k in [10,20,50,200]}
    meta['noise_sha256']=hashlib.sha256((fixtures/'initial.f32').read_bytes()).hexdigest()
    (fixtures/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
if __name__=='__main__': main()
