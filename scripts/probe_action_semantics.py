"""Video-only probes separating color changes from action-induced motion.

Optical flow is an evaluation statistic derived from RGB, never training labels.
Farneback flow on blurry predictions is only descriptive and is not ground truth.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import h5py
import numpy as np
import torch
from models.latent_actions import LatentActionModel
from scripts.bounded_picodoom import source_starts, per_image


def uniform_color_fraction(predictions):
    centered=predictions-predictions.mean(1,keepdim=True)
    color=centered.mean((-1,-2),keepdim=True)
    return color.square().mean()/centered.square().mean().clamp_min(1e-12)


def explained_fraction(feature,labels):
    residual=np.zeros_like(feature)
    for label in np.unique(labels):
        keep=labels==label;residual[keep]=feature[keep]-feature[keep].mean()
    return float(1-np.sum(residual**2)/max(np.sum((feature-feature.mean())**2),1e-12))


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True)
    p.add_argument('--name',default='action_semantics_validation')
    p.add_argument('--data',default='data/picodoom_frames.h5')
    p.add_argument('--key-file',required=True)
    p.add_argument('--output',default='results/bounded-20260905')
    args=p.parse_args();os.environ['WANDB_API_KEY']=Path(args.key_file).read_text().strip()
    import wandb
    out=Path(args.output)/args.name;out.mkdir(parents=True,exist_ok=True)
    starts=source_starts(47000,53000,2,2,128,701)
    with h5py.File(args.data) as f:
        raw=np.stack([f['frames'][[int(s),int(s+2)]] for s in starts])
    x=torch.from_numpy(raw).to('cuda',dtype=torch.float32).permute(0,1,4,2,3)/127.5-1
    flow=[]
    for pair in raw:
        g=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in pair]
        f=cv2.calcOpticalFlowFarneback(g[0],g[1],None,.5,3,15,3,5,1.2,0)
        flow.append([np.median(f[...,0]),np.median(f[...,1]),np.linalg.norm(f,axis=-1).mean()])
    flow=np.asarray(flow);features={'previous_brightness':raw[:,0].mean((1,2,3))/255,
        'brightness_delta':(raw[:,1].astype(float)-raw[:,0]).mean((1,2,3))/255,
        'median_flow_x':flow[:,0],'median_flow_y':flow[:,1],'mean_flow_magnitude':flow[:,2]}
    run=wandb.init(entity='data2yihein-d',project='tinyworlds',group='picodoom-bounded-20260905',name=args.name,
        config={'task':'evaluation only, no optimizer steps','source_starts':starts.tolist(),
                'models':args.models,'flow':'RGB-derived Farneback; descriptive only'},dir=str(out))
    all_results={};tables={}
    for entry in args.models:
        name,path=entry.split('=',1)
        model=LatentActionModel(frame_size=(64,64),patch_size=4,embed_dim=32,num_heads=8,hidden_dim=128,num_blocks=2,n_actions=4).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
        codes=torch.tensor([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]],device='cuda')
        outputs=[];ids=[];same=[]
        with torch.no_grad():
            for i in range(0,len(x),16):
                batch=x[i:i+16]
                ids.append(model.quantizer.get_indices_from_latents(model.encode(batch)).flatten())
                repeated=batch[:,0:1].expand(-1,2,-1,-1,-1)
                same.append(model.quantizer.get_indices_from_latents(model.encode(repeated)).flatten())
                outputs.append(torch.stack([model.decoder(batch,c.view(1,1,2).expand(len(batch),1,2))[:,0] for c in codes],1))
        pred=torch.cat(outputs);labels=torch.cat(ids).cpu().numpy();source_only=torch.cat(same).cpu().numpy()
        used=torch.as_tensor(np.bincount(labels,minlength=4)>0,device=pred.device)
        result={'uniform_color_fraction_of_code_change':float(uniform_color_fraction(pred)),
                'uniform_color_fraction_used_codes':float(uniform_color_fraction(pred[:,used])) if used.sum()>1 else None,
                'used_code_count':int(used.sum()),
                'same_frame_pair_code_agreement':float((labels==source_only).mean()),
                'majority_code_fraction':float(np.bincount(labels,minlength=4).max()/len(labels))}
        for feature,v in features.items():result[f'code_explained_fraction/{feature}']=explained_fraction(v,labels)
        result['per_code']={}
        for code in range(4):
            keep=labels==code
            result['per_code'][str(code)]={'count':int(keep.sum()),**{f'mean_{k}':float(v[keep].mean()) if keep.any() else None for k,v in features.items()}}
        rows=[]
        for i,s in enumerate(starts):rows.append([int(s),int(labels[i]),int(source_only[i])]+[float(features[k][i]) for k in features])
        table=wandb.Table(columns=['source_frame','inferred_code','repeated_frame_code']+list(features),data=rows)
        selected=np.linspace(0,len(x)-1,8,dtype=int)
        from PIL import Image,ImageDraw
        grid=Image.new('RGB',(768,1048),(24,24,24));draw=ImageDraw.Draw(grid)
        for j,label in enumerate(['previous','target','code 0','code 1','code 2','code 3']):draw.text((128*j+3,5),label,fill='white')
        imgs=torch.cat([x[:,0:2],pred],1)[selected]
        imgs=((imgs.cpu().clamp(-1,1)+1)*127.5).byte().permute(0,1,3,4,2).numpy()
        for i,row in enumerate(imgs):
            for j,im in enumerate(row):grid.paste(Image.fromarray(im).resize((128,128)),(128*j,24+128*i))
        filename=out/f'{name}_diverse_interventions.png';grid.save(filename)
        run.log({f'{name}/{k}':v for k,v in result.items() if isinstance(v,(float,int))},commit=False)
        run.log({f'{name}/transitions':table,f'{name}/diverse_interventions':wandb.Image(str(filename))})
        all_results[name]=result;tables[name]=rows
        print(json.dumps({name:result}),flush=True)
    (out/'metrics.json').write_text(json.dumps(all_results,indent=2))
    (out/'transitions.json').write_text(json.dumps({'columns':['source_frame','inferred_code','repeated_frame_code']+list(features),'models':tables},indent=2))
    artifact=wandb.Artifact(args.name,type='video-only-semantics-probe')
    for f in out.iterdir():
        if f.is_file():artifact.add_file(str(f))
    run.log_artifact(artifact);run.finish()


if __name__=='__main__':main()
