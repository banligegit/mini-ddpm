"""Export the final CIFAR velocity U-Net, the official 32-image examples, and CPU FP32 parity fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

PRIOR_SHA="3845447bc0d4b8c6be50d488bcfcfffebc9fc66ce5c7720b63476bbdf2ab3f75"
VAE_SHA="3477744ad62e665ea3db265f0108553d919a691f308c75db55bc8b991d231862"
STEPS=60


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_grid(src,out,mode,guidance):
    with np.load(src) as data:
        images=data["images"]
        expected=np.arange(32)%10 if mode=="cfg" else np.full(32,-1)
        if images.shape!=(32,32,32,3) or not np.array_equal(expected,data["labels"]) or int(data["steps"])!=STEPS:
            raise ValueError(f"Wrong official grid or class order: {src}")
        if int(data["seed"])!=2026 or float(data["guidance"])!=(guidance if mode=="cfg" else 0.):
            raise ValueError(f"Wrong seed/guidance for {src}")
    panel=Image.new("RGB",(8*66-2,4*66-2),"white")
    for i,img in enumerate(images):
        panel.paste(Image.fromarray(img).resize((64,64),Image.Resampling.NEAREST),((i%8)*66,(i//8)*66))
    panel.save(out,optimize=True)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task5",type=Path,required=True,help="CIFAR task 5 code and trained experiment directory")
    p.add_argument("--site",type=Path,required=True,help="Existing site/cifar-vae directory")
    args=p.parse_args()
    task=args.task5.resolve()
    diffusion=task/"prior_generation/diffusion"
    sys.path.insert(0,str(diffusion))
    from latent_unet import LatentUNet
    from cache_latents import provenance,load_gated_model
    ext=diffusion/"extension"
    selection=json.loads((task/"prior_generation/evaluation/results/guidance-validation.json").read_text("utf8"))
    if not selection["selection_final"]: raise ValueError("Validation guidance selection must be final")
    guidance=float(selection["selected_scale"])
    ck=ext/"checkpoints/ema-30000.pt"
    if digest(ck)!=PRIOR_SHA: raise ValueError("Unexpected 30k diffusion EMA")
    vae_path,channels,vae_sha=provenance()
    if vae_sha!=VAE_SHA: raise ValueError("Unexpected VAE/calibration")
    settings=json.loads((ext/"settings.json").read_text("utf8"))
    if settings["channels"]!=channels or settings["vae_sha256"]!=VAE_SHA:
        raise ValueError("Mismatched training normalization")
    torch.set_num_threads(3)
    model=LatentUNet().eval()
    saved=torch.load(ck,map_location="cpu",weights_only=True)
    if saved["step"]!=30000 or saved["vae_sha256"]!=VAE_SHA: raise ValueError("Unmatched teacher")
    model.load_state_dict(saved["ema"])
    vae=load_gated_model(vae_path,"cpu").eval()
    site=args.site/"learned-prior";fixture=site/"fixture"
    fixture.mkdir(parents=True,exist_ok=True)
    # The prior was trained on one particular VAE. Later reconstruction exports may use another.
    from export_web import Decoder, checked_export
    decoder = site / "frozen-decoder8.onnx"
    decoder_input = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(2026))
    checked_export(Decoder(vae), decoder_input, decoder, "z", "rgb")
    init=torch.randn(1,4,8,8,generator=torch.Generator().manual_seed(43001))
    ts=torch.linspace(1.,0.,STEPS+1)
    a=(ts*(np.pi/2)).cos();s=(ts*(np.pi/2)).sin()
    label=torch.tensor([7])
    graph=site/"velocity.onnx"
    torch.onnx.export(model,(init,ts[:1],label),graph,dynamo=False,opset_version=17,
                      input_names=["x","t","label"],output_names=["v"])
    options=ort.SessionOptions(); options.intra_op_num_threads=2
    engine=ort.InferenceSession(str(graph),options,providers=["CPUExecutionProvider"])
    rgb_engine=ort.InferenceSession(str(decoder),options,providers=["CPUExecutionProvider"])
    def save(name,x):
        x.detach().numpy().astype("<f4").tofile(fixture/(name+".f32"))
    def velocity(x,i,k,onnx):
        y=torch.tensor([k])
        if onnx:
            feeds={"x":x.numpy(),"t":ts[i:i+1].numpy(),"label":y.numpy()}
            return torch.from_numpy(engine.run(["v"],feeds)[0])
        return model(x,ts[i:i+1],y)
    mu=torch.tensor([c["mean"] for c in channels]).reshape(1,4,1,1)
    std=torch.tensor([c["std"] for c in channels]).reshape_as(mu)
    save("noise",init)
    errors={}
    for k in (10,7):
        mode="unconditional" if k==10 else "conditional"
        ref=velocity(init,0,k,False)
        save(f"{mode}-v",ref)
        errors[f"{mode}_v_max"]=float((ref-velocity(init,0,k,True)).abs().max())
        paths=[]
        for onnx in (False,True):
            x=init.clone()
            for i in range(STEPS):
                v=velocity(x,i,10,onnx)
                if k!=10: v=v+guidance*(velocity(x,i,k,onnx)-v)
                x0=a[i]*x-s[i]*v
                eps=s[i]*x+a[i]*v
                x=a[i+1]*x0+s[i+1]*eps
            z=x*std+mu
            rgb=torch.from_numpy(rgb_engine.run(["rgb"],{"z":z.numpy()})[0]) if onnx else vae.decode(z)
            paths.append((z,rgb))
        for (kind,ref,actual) in zip(("z","rgb"),paths[0],paths[1]):
            save(f"{mode}-{kind}",ref)
            errors[f"{mode}_{kind}_max"]=float((ref-actual).abs().max())
    if max(errors.values())>3e-4: raise RuntimeError(f"ONNX/FP32 PyTorch mismatch: {errors}")
    for mode in ("cfg","unconditional"):
        save_grid(ext/"final"/mode/"seed-2026.npz",site/(mode+".png"),mode,guidance)
    record=dict(ema_update=30000,checkpoint_sha256=PRIOR_SHA,onnx_sha256=digest(graph),
                decoder_checkpoint_sha256=VAE_SHA,decoder_onnx_sha256=digest(decoder),
                decoder_path=decoder.name,
                steps=STEPS,guidance=guidance,null_label=10,t=ts.tolist(),alpha=a.tolist(),sigma=s.tolist(),
                mean=mu.flatten().tolist(),std=std.flatten().tolist(),fixture_seed=43001,fixture_class=7,
                input_shape=[1,4,8,8],input_types={"x":"float32","t":"float32","label":"int64"},
                output_type={"v":"float32"},python_fp32_ort_max_abs=errors)
    (site/"config.json").write_text(json.dumps(record,indent=2)+"\n","utf8")
    print(json.dumps(errors),flush=True)


if __name__=="__main__": main()
