"""Cheap video-only baselines and tokenizer context-shift diagnostics."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from models.video_tokenizer import VideoTokenizer
from scripts.bounded_picodoom import source_starts


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key-file',required=True)
    p.add_argument('--data',default='data/picodoom_frames.h5')
    p.add_argument('--checkpoints',default='data/checkpoints')
    p.add_argument('--output',default='results/bounded-20260905/tokenizer_and_token_baselines')
    a=p.parse_args();torch.set_num_threads(4)
    os.environ['WANDB_API_KEY']=Path(a.key_file).read_text().strip()
    import wandb
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    run=wandb.init(entity='data2yihein-d',project='tinyworlds',group='picodoom-bounded-20260905',
        name='tokenizer_and_token_baselines',config={'task':'evaluation only','training_token_clips':2048,
        'validation_clips':128,'laplace_pseudocount':1,'alpha_selection':'training clips only','no_action_labels':True},dir=str(out))
    vt=VideoTokenizer(frame_size=(64,64),patch_size=4,embed_dim=32,num_heads=8,hidden_dim=128,num_blocks=4,latent_dim=5,num_bins=4).cuda().eval()
    ckpt=next(Path(a.checkpoints).glob('**/video_tokenizer/checkpoints/video_tokenizer_step_37500/model_state_dict.pt'))
    vt.load_state_dict(torch.load(ckpt,map_location='cuda',weights_only=True))
    with h5py.File(a.data) as f:data=torch.from_numpy(f['frames'][:])
    def clips(starts):
        idx=torch.tensor(starts[:,None]+np.arange(4)[None,:]*2)
        return data[idx].cuda().float().permute(0,1,4,2,3)/127.5-1
    counts=torch.ones(256,1024,device='cuda');train_tokens=[]
    starts=source_starts(300,46000,4,2,2048,0)
    for i in range(0,len(starts),16):
        tok=vt.tokenize(clips(starts[i:i+16]));train_tokens.append(tok)
        indices=tok[:,-1].T
        counts.scatter_add_(1,indices,torch.ones_like(indices,dtype=torch.float32))
    train_tokens=torch.cat(train_tokens)
    global_counts=torch.bincount(train_tokens[:,-1].flatten(),minlength=1024).float()+1
    marginal=global_counts/global_counts.sum();spatial=counts/counts.sum(1,keepdim=True)
    target=train_tokens[:,-1];same=(target==train_tokens[:,-2]).float();base=marginal[target]
    alpha_grid=torch.linspace(0,.99,100,device='cuda')
    losses=torch.stack([-(alpha*same+(1-alpha)*base).log().mean() for alpha in alpha_grid])
    alpha=float(alpha_grid[losses.argmin()]);result={'copy_mixture_alpha_selected_on_train':alpha,'copy_mixture_train_ce':float(losses.min())}
    for name,lo,hi in [('old_training_prefix',300,2989),('new_training_range',3000,46000),('validation',47000,53000)]:
        samples=source_starts(lo,hi,4,2,128,701);vals={}
        for i in range(0,len(samples),16):
            x=clips(samples[i:i+16]);tok=vt.tokenize(x);z=vt.quantizer.get_latents_from_indices(tok)
            shifted_tokens=vt.tokenize(x[:,1:]);shifted_z=vt.quantizer.get_latents_from_indices(shifted_tokens)
            repeated=x[:,0:1].expand(-1,4,-1,-1,-1);repeat_tokens=vt.tokenize(repeated)
            predictions={'reconstruction':vt.decoder(z)[:,-1],
                         'standalone_decode':vt.decoder(z[:,-1:])[:,0],
                         'shift_cached_decode':vt.decoder(z[:,1:])[:,-1],
                         'shift_reencoded_decode':vt.decoder(shifted_z)[:,-1],
                         'copy':x[:,-2]}
            for k,pred in predictions.items():vals.setdefault(k+'/l1',[]).append((pred-x[:,-1]).abs().mean((1,2,3)))
            vals.setdefault('token_change_after_context_shift',[]).append((tok[:,1:]!=shifted_tokens).float().mean((1,2)))
            vals.setdefault('static_input_token_change_first_to_last',[]).append((repeat_tokens[:,0]!=repeat_tokens[:,-1]).float().mean(1))
            target=tok[:,-1];same=(target==tok[:,-2]).float()
            vals.setdefault('copy_token_accuracy',[]).append(same.mean(1))
            vals.setdefault('global_unigram_ce',[]).append(-marginal[target].log().mean(1))
            vals.setdefault('spatial_unigram_ce',[]).append(-spatial[torch.arange(256,device='cuda')[None,:],target].log().mean(1))
            vals.setdefault('copy_mixture_ce',[]).append(-(alpha*same+(1-alpha)*marginal[target]).log().mean(1))
        result.update({name+'/'+k:float(torch.cat(v).mean()) for k,v in vals.items()})
    print(json.dumps(result,indent=2),flush=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    (out/'training_source_starts.json').write_text(json.dumps(starts.tolist()))
    run.log(result)
    artifact=wandb.Artifact('tokenizer-and-token-baselines',type='video-only-diagnostic')
    artifact.add_file(str(out/'metrics.json'));artifact.add_file(str(out/'training_source_starts.json'))
    run.log_artifact(artifact);run.finish()


if __name__=='__main__':main()
