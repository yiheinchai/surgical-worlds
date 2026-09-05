"""Small video-only experiments with fixed step budgets and live observability.

No environment actions, kinematics, rewards, or action labels are read. Validation
and final test are separate contiguous source ranges. The test range is opt-in.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from models.latent_actions import LatentActionModel
from models.dynamics import DynamicsModel
from models.video_tokenizer import VideoTokenizer
from models.recon_losses import reconstruction_loss


def source_starts(lo, hi, length, stride, count, seed):
    """Nonoverlapping clips, with all source indices inside [lo,hi)."""
    span = (length - 1) * stride + 1
    choices = np.arange(lo, hi - span + 1, span)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(choices, min(count, len(choices)), replace=False))


def per_image(pred, target, prev):
    dims = tuple(range(1, pred.ndim))
    err = (pred.float() - target.float()).abs()
    motion = (target.float() - prev.float()).abs().mean(1, keepdim=True)
    weight = motion.expand_as(err)
    moving = (motion > 0.15).expand_as(err)
    return {
        'l1': err.mean(dims),
        'motion_weighted_l1': (err * weight).sum(dims) / weight.sum(dims).clamp_min(1e-6),
        'moving_region_l1': (err * moving).sum(dims) / moving.sum(dims).clamp_min(1),
        'objective': F.smooth_l1_loss(pred.float(), target.float(), reduction='none').mean(dims)
        + .15 * ((pred.float().diff(dim=-1)-target.float().diff(dim=-1)).abs().mean(dims)
                 + (pred.float().diff(dim=-2)-target.float().diff(dim=-2)).abs().mean(dims)),
    }


def code_stats(z, q, quantizer):
    ids = quantizer.get_indices_from_latents(q).flatten()
    counts = torch.bincount(ids, minlength=4).float()
    p = counts / counts.sum().clamp_min(1)
    out = {f'code_{i}_fraction': float(p[i]) for i in range(4)}
    out.update(entropy_nats=float(-(p * p.clamp_min(1e-12).log()).sum()),
               unique_codes=int((counts > 0).sum()),
               encoder_variance=float(z.flatten(0, -2).var(0, unbiased=False).mean()),
               encoder_abs_mean=float(z.abs().mean()),
               encoder_saturation_fraction=float((z.abs() > 3).float().mean()))
    for i in range(z.shape[-1]):
        out[f'encoder_dim_{i}_positive_fraction'] = float((z[..., i] > 0).float().mean())
        out[f'encoder_dim_{i}_std'] = float(z[..., i].std(unbiased=False))
    return out


def soft_code_information(z):
    """Diagnostic-only differentiable mutual-information proxy for binary FSQ."""
    p = (z.tanh() + 1) / 2
    joint = torch.stack([(1-p[..., 0])*(1-p[..., 1]), p[..., 0]*(1-p[..., 1]),
                         (1-p[..., 0])*p[..., 1], p[..., 0]*p[..., 1]], -1).flatten(0, -2)
    marginal = joint.mean(0)
    entropy = -(marginal * marginal.clamp_min(1e-8).log()).sum()
    conditional = -(joint * joint.clamp_min(1e-8).log()).sum(-1).mean()
    return conditional - entropy


def lam_encode(lam, clips, pairwise=False):
    if not pairwise:
        return lam.encode(clips)
    b, t, c, h, w = clips.shape
    pairs = torch.stack([clips[:, :-1], clips[:, 1:]], 2).reshape(-1, 2, c, h, w)
    return lam.encode(pairs).reshape(b, t-1, -1)


class Experiment:
    def __init__(self, args):
        self.a = args
        self.device = torch.device(args.device)
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
        self.start = time.monotonic()
        self.out = Path(args.output) / args.name
        self.out.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        if args.wandb_key_file:
            os.environ['WANDB_API_KEY'] = Path(args.wandb_key_file).read_text().strip()
        config = vars(args).copy()
        config.pop('wandb_key_file')
        config.update(stopping='fixed optimizer steps; no validation early stopping',
                      max_wall_seconds=args.max_seconds, data_supervision='RGB video only',
                      precision='float32 with TF32 on CUDA', gradient_accumulation_steps=1,
                      source_ranges={'train':[300,46000], 'validation':[47000,53000], 'test':[54000,59785]})
        self.config = config
        (self.out/'config.json').write_text(json.dumps(config, indent=2))
        if args.wandb_mode != 'disabled':
            import wandb
            self.wandb = wandb
            self.run = wandb.init(entity='data2yihein-d', project='tinyworlds',
                group='picodoom-bounded-20260905', name=args.name, config=config,
                dir=str(self.out), mode=args.wandb_mode,
                settings=wandb.Settings(init_timeout=90))
            print('WANDB_URL', self.run.url, flush=True)
        with h5py.File(args.data,'r') as f:
            self.data = torch.from_numpy(f['frames'][:])
        assert self.data.ndim == 4 and tuple(self.data.shape[1:]) == (64,64,3), self.data.shape
        assert len(self.data) >= 59785
        self.length = 2 if args.stage == 'lam' else 4
        self.starts = {
            'train':source_starts(300,46000,self.length,2,args.train_clips,args.seed),
            'validation':source_starts(47000,53000,self.length,2,args.eval_clips,701),
            'test':source_starts(54000,59785,self.length,2,args.eval_clips,702),
        }
        (self.out/'source_starts.json').write_text(json.dumps({k:v.tolist() for k,v in self.starts.items()},indent=2))
        common = dict(frame_size=(64,64),patch_size=4,embed_dim=32,num_heads=8,hidden_dim=128)
        self.lam = LatentActionModel(**common,num_blocks=2,n_actions=4).to(self.device)
        self.vt = VideoTokenizer(**common,num_blocks=4,latent_dim=5,num_bins=4).to(self.device)
        self.dyn = DynamicsModel(**common,num_blocks=8,latent_dim=5,num_bins=4,conditioning_dim=2).to(self.device)
        self.load(self.vt, 'video_tokenizer', 'video_tokenizer_step_37500')
        if args.init == 'checkpoint' or args.stage == 'dynamics':
            self.load(self.lam, 'latent_actions', 'latent_actions_step_9500')
        if args.stage == 'dynamics':
            self.load(self.dyn,'dynamics','dynamics_step_44000')
        if args.lam_weights:
            self.lam.load_state_dict(torch.load(args.lam_weights,map_location=self.device,weights_only=True))
        if args.dynamics_weights:
            self.dyn.load_state_dict(torch.load(args.dynamics_weights,map_location=self.device,weights_only=True))
        self.vt.eval().requires_grad_(False)
        if args.stage == 'lam':
            self.model=self.lam
        else:
            self.lam.eval().requires_grad_(False)
            self.model=self.dyn
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=args.lr,weight_decay=.01)
        if self.wandb:
            self.run.watch(self.model, log='all', log_freq=max(args.log_every, 100))
        self.generator=torch.Generator().manual_seed(args.seed+100)
        self.log({'parameters':sum(p.numel() for p in self.model.parameters()),
                  'train_clips':len(self.starts['train']), 'device_name':torch.cuda.get_device_name() if self.device.type=='cuda' else 'cpu'},0)
        self.cache={}
        if args.stage=='dynamics':
            for split in ('train','validation') + (('test',) if args.test else ()):
                chunks=[]; ac=[]
                for i in range(0,len(self.starts[split]),args.batch):
                    x=self.clips(split,slice(i,i+args.batch))
                    with torch.no_grad():
                        chunks.append(self.vt.tokenize(x).cpu())
                        ac.append(lam_encode(self.lam,x,args.pairwise_actions).cpu())
                self.cache[split]=(torch.cat(chunks),torch.cat(ac))
                print('CACHED',split,len(chunks),flush=True)

    def load(self, model, stage, name):
        root=Path(self.a.checkpoints)
        matches=list(root.glob(f'**/{stage}/checkpoints/{name}/model_state_dict.pt'))
        assert len(matches)==1,(stage,name,matches)
        model.load_state_dict(torch.load(matches[0],map_location='cpu',weights_only=True))

    def clips(self, split, indices):
        starts=torch.as_tensor(self.starts[split][indices].copy())
        idx=starts[:,None]+torch.arange(self.length)[None,:]*2
        return self.data[idx].to(self.device,dtype=torch.float32).permute(0,1,4,2,3)/127.5-1

    def log(self, metrics, step):
        record={'step':step,'elapsed_seconds':time.monotonic()-self.start,**metrics}
        with (self.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if self.wandb:self.run.log(record,step=step,commit=False)
        print(json.dumps(record),flush=True)

    def images(self, name, columns, labels, step):
        # Rows are clips, columns are explicitly labelled counterfactuals/baselines.
        from PIL import Image, ImageDraw
        arrays=[((x[:8].detach().cpu().clamp(-1,1)+1)*127.5).byte().permute(0,2,3,1).numpy() for x in columns]
        canvas=Image.new('RGB',(128*len(arrays),24+128*len(arrays[0])),(28,28,28));d=ImageDraw.Draw(canvas)
        for c,(arr,label) in enumerate(zip(arrays,labels)):
            d.text((c*128+3,5),label,fill='white')
            for r,im in enumerate(arr):canvas.paste(Image.fromarray(im).resize((128,128)),(128*c,24+128*r))
        path=self.out/f'{name}_{step:06d}.png';canvas.save(path)
        if self.wandb:self.run.log({name:self.wandb.Image(str(path))},step=step,commit=False)

    @torch.no_grad()
    def evaluate_lam(self, split, step):
        self.lam.eval()
        values={}; zs=[];qs=[];first=None
        all_x=self.clips(split,slice(None))
        codes=torch.tensor([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]],device=self.device)
        for i in range(0,len(all_x),self.a.batch):
            x=all_x[i:i+self.a.batch];z=self.lam.encoder(x);q=self.lam.quantizer(z)
            zs.append(z);qs.append(q)
        all_q=torch.cat(qs);all_z=torch.cat(zs)
        ids=self.lam.quantizer.get_indices_from_latents(all_q).flatten()
        mode=int(torch.bincount(ids,minlength=4).argmax())
        # Fixed cyclic permutation; records unchanged-code fraction under collapse.
        shuffled=all_q.roll(1,0)
        for i in range(0,len(all_x),self.a.batch):
            x=all_x[i:i+self.a.batch];q=all_q[i:i+self.a.batch];sh=shuffled[i:i+self.a.batch]
            preds={'copy':x[:,0],
                   'inferred':self.lam.decoder(x,q)[:,0],
                   'shuffled':self.lam.decoder(x,sh)[:,0],
                   'constant':self.lam.decoder(x,codes[mode].view(1,1,2).expand_as(q))[:,0]}
            choices=torch.stack([self.lam.decoder(x,c.view(1,1,2).expand_as(q))[:,0] for c in codes],1)
            oracle_losses=torch.stack([per_image(choices[:,j],x[:,1],x[:,0])['objective'] for j in range(4)],1)
            oracle_ids=oracle_losses.argmin(1)
            preds['oracle']=choices[torch.arange(len(x),device=self.device),oracle_ids]
            for name,pred in preds.items():
                for metric,v in per_image(pred,x[:,1],x[:,0]).items():values.setdefault(f'{name}/{metric}',[]).append(v.cpu())
            values.setdefault('oracle_encoder_agreement',[]).append((oracle_ids==ids[i:i+len(x)]).float().cpu())
            sensitivity=(choices-choices.mean(1,keepdim=True)).abs().mean((1,2,3,4))
            values.setdefault('intervention_spread_l1',[]).append(sensitivity.cpu())
            if first is None:first=(x,preds,choices)
        full={k:torch.cat(v) for k,v in values.items()}
        metrics={f'{split}/{k}':float(v.mean()) for k,v in full.items()}
        for metric in ('l1','motion_weighted_l1','objective'):
            for baseline in ('shuffled','constant','copy'):
                diff=full[f'{baseline}/{metric}']-full[f'inferred/{metric}']
                metrics[f'{split}/gap_{baseline}_minus_inferred/{metric}']=float(diff.mean())
                metrics[f'{split}/gap_{baseline}_minus_inferred/{metric}_se']=float(diff.std(unbiased=False)/math.sqrt(len(diff)))
        metrics.update({f'{split}/actions/{k}':v for k,v in code_stats(all_z,all_q,self.lam.quantizer).items()})
        metrics[f'{split}/shuffled_code_changed_fraction']=float((shuffled!=all_q).any(-1).float().mean())
        metrics[f'{split}/constant_code']=mode
        self.log(metrics,step)
        if self.wandb:
            histogram=np.bincount(ids.cpu().numpy(),minlength=4)
            self.run.log({f'{split}/action_distribution':self.wandb.Histogram(
                np_histogram=(histogram,np.arange(5)))},step=step,commit=False)
        x,pred,choices=first
        self.images(f'{split}/predictions'.replace('/','_'),[x[:,0],x[:,1],pred['inferred'],pred['shuffled'],pred['oracle']],['previous','target','inferred','shuffled','oracle (target)'],step)
        self.images(f'{split}_interventions',[x[:,0]]+[choices[:,j] for j in range(4)],['previous','code 0','code 1','code 2','code 3'],step)
        return metrics

    @torch.no_grad()
    def evaluate_dynamics(self, split, step):
        self.dyn.eval();ts,ac=self.cache[split]
        shuffled=ac.roll(1,0)
        ids=self.lam.quantizer.get_indices_from_latents(ac).flatten()
        mode=int(torch.bincount(ids,minlength=4).argmax())
        code=self.lam.quantizer.get_latents_from_indices(torch.tensor(mode,device=self.device))
        vals={};first=None
        for i in range(0,len(ts),self.a.batch):
            x=self.clips(split,slice(i,i+self.a.batch));tokens=ts[i:i+self.a.batch].to(self.device)
            lat=self.vt.quantizer.get_latents_from_indices(tokens)
            acts=ac[i:i+self.a.batch].to(self.device)
            preds={}
            for name,actions in [('inferred',acts),('shuffled',shuffled[i:i+self.a.batch].to(self.device)),('constant',code.view(1,1,-1).expand_as(acts))]:
                logits,_,_=self.dyn(lat,conditioning=actions,targets=tokens,objective_mode='next_frame')
                ce=F.cross_entropy(logits[:,-1].flatten(0,1),tokens[:,-1].flatten(),reduction='none').reshape(len(x),-1).mean(1)
                vals.setdefault(f'{name}/fully_hidden_ce',[]).append(ce.cpu())
                devices=[self.device.index or torch.cuda.current_device()] if self.device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(1000+i)
                    partial,mask,_=self.dyn(lat,conditioning=actions,targets=tokens,objective_mode='legacy_maskgit')
                lp=F.cross_entropy(partial.flatten(0,2),tokens.flatten(),reduction='none').reshape_as(tokens)
                masked_mean=(lp*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
                vals.setdefault(f'{name}/partial_reconstruction_ce',[]).append(masked_mean.cpu())
                pred_lat=self.vt.quantizer.get_latents_from_indices(logits[:,-1:].argmax(-1))
                # Decode WITH causal history, as the spatiotemporal tokenizer was trained.
                pred=self.vt.decoder(torch.cat([lat[:,:-1],pred_lat],1))[:,-1]
                preds[name]=pred
                if name=='inferred':
                    alone=self.vt.decoder(pred_lat)[:,0]
                    vals.setdefault('inferred/standalone_decode_l1',[]).append((alone-x[:,-1]).abs().mean((1,2,3)).cpu())
                for k,v in per_image(pred,x[:,-1],x[:,-2]).items():vals.setdefault(f'{name}/{k}',[]).append(v.cpu())
            for name,pred in [('copy',x[:,-2]),('tokenizer',self.vt.decoder(lat)[:,-1])]:
                for k,v in per_image(pred,x[:,-1],x[:,-2]).items():vals.setdefault(f'{name}/{k}',[]).append(v.cpu())
            if first is None:first=(x,lat,acts,preds)
        full={k:torch.cat(v) for k,v in vals.items()}
        metrics={f'{split}/{k}':float(v.mean()) for k,v in full.items()}
        counts=torch.bincount(ids,minlength=4).float();p=counts/counts.sum()
        metrics.update({f'{split}/actions/code_{j}_fraction':float(p[j]) for j in range(4)})
        metrics[f'{split}/actions/entropy_nats']=float(-(p*p.clamp_min(1e-12).log()).sum())
        metrics[f'{split}/actions/unique_codes']=int((counts>0).sum())
        metrics[f'{split}/shuffled_code_changed_fraction']=float((shuffled!=ac).any(-1).float().mean())
        for metric in ('fully_hidden_ce','l1','motion_weighted_l1'):
            for base in ('shuffled','constant'):
                d=full[f'{base}/{metric}']-full[f'inferred/{metric}']
                metrics[f'{split}/gap_{base}_minus_inferred/{metric}']=float(d.mean())
                metrics[f'{split}/gap_{base}_minus_inferred/{metric}_se']=float(d.std(unbiased=False)/math.sqrt(len(d)))
        self.log(metrics,step)
        x,lat,act,pred=first
        self.images(f'{split}_predictions',[x[:,-2],x[:,-1],pred['inferred'],pred['shuffled']],['previous','target','inferred','shuffled'],step)
        if self.a.rollouts and (step==0 or step==self.a.steps):self.rollout(split,step)
        return metrics

    @torch.no_grad()
    def rollout(self,split,step):
        # Future-video-inferred actions are oracle diagnostic inputs;
        # generated history receives no future frame pixels or future tokens.
        starts=self.starts[split][:4];idx=torch.as_tensor(starts[:,None]+np.arange(19)[None,:]*2)
        x=self.data[idx].to(self.device,dtype=torch.float32).permute(0,1,4,2,3)/127.5-1
        # Infer with fixed four-frame windows so LAM context matches training.
        actions=[]
        for j in range(3,19):actions.append(lam_encode(self.lam,x[:,j-3:j+1],self.a.pairwise_actions)[:,-1])
        seed_lat=self.vt.quantizer.get_latents_from_indices(self.vt.tokenize(x[:,:3]))
        hist_actions=lam_encode(self.lam,x[:,:3],self.a.pairwise_actions)
        rendered={};metrics={}
        for name in ('inferred','constant'):
            lat=seed_lat.clone();past=hist_actions.clone();frames=[]
            for j in range(16):
                action=actions[j] if name=='inferred' else actions[0]
                past=torch.cat([past,action[:,None]],1)[:,-3:]
                masked=torch.cat([lat[:,-3:],self.dyn.mask_token.expand(len(x),1,lat.shape[2],-1)],1)
                logits,_,_=self.dyn(masked,training=False,conditioning=past)
                nxt=self.vt.quantizer.get_latents_from_indices(logits[:,-1:].argmax(-1))
                lat=torch.cat([lat[:,-3:],nxt],1)
                frames.append(self.vt.decoder(lat)[:,-1])
            pred=torch.stack(frames,1);rendered[name]=pred
            for h in (1,4,8,16):metrics[f'{split}/rollout_{name}/l1_at_{h}']=float((pred[:,h-1]-x[:,h+2]).abs().mean())
        self.log(metrics,step)
        # Video panels: ground truth | repeated initial frame | inferred | constant.
        v=torch.cat([x[:,3:],x[:,2:3].expand(-1,16,-1,-1,-1),rendered['inferred'],rendered['constant']],-1)
        v=torch.cat(list(v),-2)
        arr=((v.cpu().clamp(-1,1)+1)*127.5).byte().numpy()
        np.save(self.out/f'{split}_rollout_{step:06d}.npy',arr)
        from PIL import Image
        gif=self.out/f'{split}_rollout_{step:06d}.gif'
        frames=[Image.fromarray(f.transpose(1,2,0)) for f in arr]
        frames[0].save(gif,save_all=True,append_images=frames[1:],duration=200,loop=0)
        if self.wandb:self.run.log({f'{split}/rollout_video':self.wandb.Video(str(gif),caption='Rows: clips. Columns: target | copy | inferred video actions | constant. No action labels.')},step=step,commit=False)

    def train(self):
        evaluate=self.evaluate_lam if self.a.stage=='lam' else self.evaluate_dynamics
        best=math.inf;beststep=0
        evaluate('validation',0)
        start=time.monotonic();prev=start;losses=[]
        completed=0;status='completed_fixed_steps'
        for step in range(1,self.a.steps+1):
            if time.monotonic()-self.start>self.a.max_seconds:
                status='interrupted_wall_time_cap';break
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            ids=torch.randint(len(self.starts['train']),(self.a.batch,),generator=self.generator)
            if self.a.stage=='lam':
                x=self.clips('train',ids.numpy());z=self.lam.encoder(x);q=self.lam.quantizer(z)
                pred=self.lam.decoder(x,q)
                recon=reconstruction_loss(pred,x[:,1:])
                var=z.var(dim=0,unbiased=False).mean();vp=100*F.relu(.01-var)
                aux=soft_code_information(z)*self.a.information_weight
                loss=recon+vp+aux
                extra={'train/reconstruction':float(recon.detach()),'train/variance_penalty':float(vp.detach()),'train/information_penalty':float(aux.detach())}
            else:
                ts,ac=self.cache['train'];tok=ts[ids].to(self.device);act=ac[ids].to(self.device)
                lat=self.vt.quantizer.get_latents_from_indices(tok)
                _,_,loss=self.dyn(lat,conditioning=act,targets=tok,objective_mode=self.a.objective)
                extra={}
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
            loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            if not torch.isfinite(gn):raise FloatingPointError(f'Nonfinite gradient at {step}')
            self.optimizer.step();completed=step;losses.append(float(loss.detach()))
            if step==1 or step%self.a.log_every==0:
                if self.device.type=='cuda':torch.cuda.synchronize()
                elapsed=time.monotonic()-prev
                log={'train/loss':float(np.mean(losses)),'train/gradient_norm':float(gn),
                     'perf/optimizer_steps_per_second':len(losses)/max(elapsed,1e-6),
                     'perf/cuda_peak_allocated_gb':torch.cuda.max_memory_allocated()/1e9 if self.device.type=='cuda' else 0,**extra}
                if self.a.stage=='lam':log.update({f'train/actions/{k}':v for k,v in code_stats(z.detach(),q.detach(),self.lam.quantizer).items()})
                self.log(log,step);losses=[];prev=time.monotonic()
            if step%self.a.eval_every==0 or step==self.a.steps:
                metrics=evaluate('validation',step)
                key='validation/inferred/objective' if self.a.stage=='lam' else 'validation/inferred/fully_hidden_ce'
                if metrics[key]<best:
                    best=metrics[key];beststep=step
                    torch.save(self.model.state_dict(),self.out/'best_weights.pt')
                torch.save(self.model.state_dict(),self.out/'last_weights.pt')
                prev=time.monotonic()
        if completed==0 or completed%self.a.eval_every!=0 and completed!=self.a.steps:
            evaluate('validation',completed)
        torch.save(self.model.state_dict(),self.out/'last_weights.pt')
        if self.a.test:evaluate('test',completed)
        result={'status':status,'optimizer_steps_completed':completed,'best_validation_step':beststep,
                'best_validation_metric':best if math.isfinite(best) else None,'elapsed_seconds':time.monotonic()-self.start}
        (self.out/'result.json').write_text(json.dumps(result,indent=2))
        self.log(result,completed)
        if self.wandb:
            artifact=self.wandb.Artifact(self.a.name,type='bounded-experiment')
            for f in self.out.iterdir():
                if f.is_file() and f.suffix in ('.json','.jsonl','.pt','.png','.gif'):artifact.add_file(str(f))
            self.run.log_artifact(artifact);self.run.finish()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['lam','dynamics'],required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--data',default='data/picodoom_frames.h5')
    p.add_argument('--checkpoints',default='data/checkpoints')
    p.add_argument('--output',default='results/bounded-20260905')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--max-seconds',type=int,default=1800)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--train-clips',type=int,default=4096)
    p.add_argument('--eval-clips',type=int,default=128)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--log-every',type=int,default=50)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--init',choices=['scratch','checkpoint'],default='checkpoint')
    p.add_argument('--objective',choices=['legacy_maskgit','next_frame'],default='next_frame')
    p.add_argument('--information-weight',type=float,default=0.)
    p.add_argument('--lam-weights')
    p.add_argument('--dynamics-weights')
    p.add_argument('--pairwise-actions',action='store_true')
    p.add_argument('--test',action='store_true')
    p.add_argument('--rollouts',action='store_true')
    p.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    p.add_argument('--wandb-key-file')
    a=p.parse_args()
    assert a.steps>=0 and a.max_seconds>0 and a.batch>0 and a.eval_every>0
    try:
        Experiment(a).train()
    except Exception as exc:
        out=Path(a.output)/a.name;out.mkdir(parents=True,exist_ok=True)
        (out/'failure.json').write_text(json.dumps({'status':'failed','error_type':type(exc).__name__,'message':str(exc)},indent=2))
        if a.wandb_mode!='disabled':
            import wandb
            if wandb.run is not None:wandb.run.finish(exit_code=1)
        raise


if __name__=='__main__':main()
