"""Evaluate synthetic code/translation consistency; labels never train a model."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from models.latent_actions import LatentActionModel
from scripts.synthetic_branched_motion import translated_pairs


def code_conditional_entropy(labels):
    counts=torch.stack([(labels==j).float().mean(-1) for j in range(4)],-1)
    return float(-(counts*counts.clamp_min(1e-12).log()).sum(-1).mean())


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key-file',required=True)
    p.add_argument('--prefix',default='synthetic_branched');p.add_argument('--name',default='synthetic_code_semantics');a=p.parse_args()
    torch.set_num_threads(4)
    os.environ['WANDB_API_KEY']=Path(a.key_file).read_text().strip()
    import wandb
    name=a.name;out=Path('results/bounded-20260905')/name;out.mkdir(parents=True,exist_ok=True)
    run=wandb.init(entity='data2yihein-d',project='tinyworlds',group='picodoom-bounded-20260905',name=name,
        config={'training':'none','model_prefix':a.prefix,'labels':'known synthetic translation index for evaluation only',
                'permutation_selection':'training textures only, evaluate on unseen validation textures',
                'chance_accuracy':.25},dir=str(out))
    with h5py.File('data/picodoom_frames.h5') as f:data=torch.from_numpy(f['frames'][:])
    results={}
    for variant in ('reference','information'):
        model=LatentActionModel(frame_size=(64,64),patch_size=4,embed_dim=32,num_heads=8,hidden_dim=128,num_blocks=2,n_actions=4).cuda().eval()
        model.load_state_dict(torch.load(f'results/bounded-20260905/{a.prefix}_{variant}/last_weights.pt',map_location='cuda',weights_only=True))
        predictions={}
        for split,lo,hi,count in [('train',3000,46000,64),('validation',47000,53000,32)]:
            textures=data[np.linspace(lo,hi-1,count,dtype=int)].float().permute(0,3,1,2)/127.5-1
            pairs=translated_pairs(textures).cuda();labels=[]
            for i in range(0,len(pairs),16):labels.append(model.quantizer.get_indices_from_latents(model.encode(pairs[i:i+16])).cpu())
            predictions[split]=torch.cat(labels).reshape(count,4)
        candidates=list(itertools.permutations(range(4)));target=torch.arange(4)[None,:]
        scores=[float((torch.tensor(mapping)[predictions['train']]==target).float().mean()) for mapping in candidates]
        mapping=candidates[int(np.argmax(scores))]
        result={'train_selected_code_to_translation':mapping,'train_mapping_accuracy':max(scores),
                'validation_mapping_accuracy':float((torch.tensor(mapping)[predictions['validation']]==target).float().mean()),
                'validation_code_entropy_given_texture':code_conditional_entropy(predictions['validation']),
                'validation_all_branches_same_code_fraction':float((predictions['validation']==predictions['validation'][:,:1]).all(1).float().mean())}
        results[variant]=result
        run.log({variant+'/'+k:v for k,v in result.items() if isinstance(v,float)})
    (out/'metrics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
    artifact=wandb.Artifact(name,type='video-only-diagnostic');artifact.add_file(str(out/'metrics.json'));run.log_artifact(artifact);run.finish()


if __name__=='__main__':main()
