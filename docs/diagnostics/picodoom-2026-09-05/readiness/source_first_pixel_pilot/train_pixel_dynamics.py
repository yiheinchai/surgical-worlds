"""Bounded, observable RGB diffusion experiment conditioned on unsupervised codes.

Inference accepts history and integer latent controls only. Evaluation-inferred
controls use future RGB and are explicitly oracle controls, not autonomous play.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,hashlib,json,os,time
import h5py,numpy as np,torch
from PIL import Image
from models.pixel_diffusion import PixelDenoiser,denoising_loss
from models.motion_codes import rgb_flow,flow_descriptor,assign_codes,code_metrics


def image_array(x):return ((x.detach().float().clamp(-1,1)+1)*127.5).byte().permute(0,2,3,1).cpu().numpy()

class Pilot:
    def __init__(self,a):
        self.a=a;self.out=Path(a.output)/a.name;self.out.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        self.device=torch.device(a.device)
        with h5py.File(a.data,'r') as f:self.data=torch.from_numpy(f['frames'][:])
        codes=np.load(a.codes);self.ids=torch.from_numpy(codes['ids'].astype(np.int64));self.centers=codes['centers'];self.stride=int(codes['stride']);self.history=4
        span=self.stride*4+1
        self.train_starts=np.arange(300,46000-span+1);self.val=np.linspace(47000,53000-span-1,a.eval_clips,dtype=int)
        self.rng=np.random.default_rng(a.seed)
        self.model=PixelDenoiser(len(self.centers),4,a.width).to(self.device);self.ema=copy.deepcopy(self.model).eval();self.opt=torch.optim.AdamW(self.model.parameters(),lr=a.lr,weight_decay=.01)
        self.start_step=0
        if a.resume:
            ck=torch.load(a.resume,map_location=self.device,weights_only=True);self.model.load_state_dict(ck['model']);self.ema.load_state_dict(ck['ema']);self.opt.load_state_dict(ck['optimizer']);self.start_step=ck['step']
        self.config={**vars(a),'supervision':'RGB-only classical flow plus train-only learned k-means codes','history':4,'stride':self.stride,'parameters':sum(p.numel() for p in self.model.parameters()),'train_range':[300,46000],'validation_range':[47000,53000],'validation_reused':True,'validation_controls':'future-RGB-inferred oracle for paired metrics; direct integer interventions separately','git_head':os.popen('git rev-parse HEAD').read().strip(),'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['models/pixel_diffusion.py','models/motion_codes.py','scripts/train_pixel_dynamics.py']}}
        (self.out/'config.json').write_text(json.dumps(self.config,indent=2));(self.out/'validation_starts.json').write_text(json.dumps(self.val.tolist()))
        self.run=None;self.wb=None
        if a.key_file:
            os.environ['WANDB_API_KEY']=Path(a.key_file).read_text().strip()
            import wandb
            self.wb=wandb;self.run=wandb.init(entity='data2yihein-d',project='tinyworlds',group='picodoom-readiness-20260905',name=a.name,config=self.config,dir=str(self.out),mode=a.wandb_mode,settings=wandb.Settings(init_timeout=60))
            self.run.watch(self.model,log='all',log_freq=1000,log_graph=False);print('W&B',self.run.url,flush=True)
            for split,indices in [('train',self.train_starts),('validation',self.val)]:
                stats=code_metrics(self.ids[torch.as_tensor(indices)].numpy(),len(self.centers))
                self.run.log({f'codes/{split}_entropy':stats['entropy_nats'],**{f'codes/{split}_{k}_fraction':v for k,v in enumerate(stats['fractions'])}})
        self.started=time.time()
    def clips(self,starts):
        ix=torch.as_tensor(starts)[:,None]+torch.arange(5)[None]*self.stride
        x=self.data[ix].to(self.device).float().permute(0,1,4,2,3)/127.5-1
        act=self.ids[ix[:,:-1]].to(self.device)
        if self.a.conditioning=='constant':act=torch.zeros_like(act)
        return x[:,:4],x[:,4],act
    def log(self,step,row):
        row={'step':step,'elapsed_seconds':time.time()-self.started,**row}
        with (self.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if self.run:self.run.log(row)
        print(json.dumps(row),flush=True)
    @torch.no_grad()
    def evaluate(self,step):
        sums={};saved=[];predids=[];trueids=[]
        for j in range(0,len(self.val),16):
            h,y,act=self.clips(self.val[j:j+16]);pred=self.ema.sample(h,act,self.a.sample_steps,seed=700+j)
            shuffled=act.clone();shuffled[:,-1]=act[:,-1].roll(1)
            wrong=self.ema.sample(h,shuffled,self.a.sample_steps,seed=700+j)
            constant=act.clone();constant[:,-1]=0
            fixed=self.ema.sample(h,constant,self.a.sample_steps,seed=700+j)
            motion=(y-h[:,-1]).abs().mean(1,keepdim=True)
            for name,p in [('inferred',pred),('shuffled',wrong),('constant',fixed),('copy',h[:,-1])]:
                err=(p-y).abs();vals={'l1':err.mean((1,2,3)),'mse':((p-y)**2).mean((1,2,3)),'motion_l1':(err*motion).sum((1,2,3))/(motion.expand_as(err).sum((1,2,3)).clamp_min(1e-6))}
                for k,v in vals.items():sums.setdefault(f'validation/{name}_{k}',[]).extend(v.cpu().tolist())
            prev_np,pred_np=image_array(h[:,-1]),image_array(pred)
            ds=np.stack([flow_descriptor(rgb_flow(a,b)) for a,b in zip(prev_np,pred_np)])
            predids.extend(assign_codes(ds,self.centers).tolist());trueids.extend(act[:,-1].cpu().tolist())
            if j==0:saved=[h[:8,-1],y[:8],pred[:8],wrong[:8],fixed[:8]]
        row={k:float(np.mean(v)) for k,v in sums.items()};row['validation/shuffle_l1_gap']=row['validation/shuffled_l1']-row['validation/inferred_l1'];row['validation/generated_motion_code_agreement']=float((np.array(predids)==trueids).mean())
        row['validation/generated_motion_entropy']=code_metrics(predids,len(self.centers))['entropy_nats']
        # Per-sample matched differences support later bootstrap uncertainty.
        (self.out/f'evaluation_{step:06}.json').write_text(json.dumps({'means':row,'per_sample':sums,'generated_motion_ids':predids,'requested_ids':trueids},indent=2))
        tile=np.concatenate([np.concatenate(image_array(t),axis=1) for t in saved],axis=0);path=self.out/f'predictions_{step:06}.png';Image.fromarray(tile).resize((1024,640),Image.Resampling.NEAREST).save(path)
        if self.run:self.run.log({'samples/previous_target_inferred_shuffled_constant':self.wb.Image(str(path)),'step':step})
        self.intervene(step);self.log(step,row);return row
    @torch.no_grad()
    def intervene(self,step):
        starts=np.array([47000,48500,50000,51500]);h,_,a=self.clips(starts);rows=[];measured=[]
        for k in range(len(self.centers)):
            act=a.clone();act[:,-1]=k;p=self.ema.sample(h,act,self.a.sample_steps,seed=808)
            rows.append(np.concatenate(image_array(p),axis=1))
            ds=np.stack([flow_descriptor(rgb_flow(x,y)) for x,y in zip(image_array(h[:,-1]),image_array(p))]);measured.append(ds.tolist())
        Image.fromarray(np.concatenate(rows,0)).resize((512,len(rows)*128),Image.Resampling.NEAREST).save(self.out/f'interventions_{step:06}.png')
        (self.out/f'interventions_{step:06}.json').write_text(json.dumps({'source_starts':starts.tolist(),'generated_flow_descriptors_by_requested_code':measured}))
        if self.run:self.run.log({'samples/direct_code_interventions':self.wb.Image(str(self.out/f'interventions_{step:06}.png')),'step':step})
    def checkpoint(self,step):
        state={'model':self.model.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.opt.state_dict(),'step':step}
        torch.save(state,self.out/'last.pt');torch.save({'model':self.ema.state_dict(),'width':self.a.width,'codes':len(self.centers),'history':4,'step':step},self.out/f'ema_{step:06}.pt')
    def train(self):
        interval=[];last=time.time();status='finished';step=self.start_step
        try:
            for step in range(self.start_step+1,self.a.steps+1):
                if time.time()-self.started>self.a.max_seconds:status='time_limit';break
                ix=self.rng.choice(self.train_starts,self.a.batch);h,y,act=self.clips(ix);self.opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                    loss,detail=denoising_loss(self.model,h,y,act,self.a.context_noise)
                assert torch.isfinite(loss),'Nonfinite loss';loss.backward();gn=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);assert torch.isfinite(gn),'Nonfinite gradients';self.opt.step()
                with torch.no_grad():
                    for ep,p in zip(self.ema.parameters(),self.model.parameters()):ep.lerp_(p,1-min(.995,(step+1)/(step+10)))
                interval.append(float(loss.detach()))
                if step%self.a.log_every==0:
                    now=time.time();self.log(step,{'train/loss':float(np.mean(interval)),'train/gradient_norm':float(gn),'train/updates_per_second':len(interval)/(now-last),'train/peak_memory_gb':torch.cuda.max_memory_allocated()/1e9 if self.device.type=='cuda' else 0,**{'train/'+k:v for k,v in detail.items()}});interval=[];last=now
                if step%self.a.eval_every==0 or step==self.a.steps:
                    self.checkpoint(step);self.evaluate(step);last=time.time()
        except Exception:
            status='failed';raise
        finally:
            self.checkpoint(step);(self.out/'status.json').write_text(json.dumps({'status':status,'step':step,'elapsed_seconds':time.time()-self.started}));
            if self.run:self.run.finish(exit_code=0 if status!='failed' else 1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--data',default='data/picodoom_frames.h5');p.add_argument('--codes',default='results/readiness/motion/motion_codes.npz');p.add_argument('--output',default='results/readiness');p.add_argument('--device',default='cuda');p.add_argument('--key-file');p.add_argument('--wandb-mode',default='online',choices=['online','offline']);p.add_argument('--steps',type=int,default=5000);p.add_argument('--max-seconds',type=int,default=2400);p.add_argument('--batch',type=int,default=32);p.add_argument('--width',type=int,default=32);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--seed',type=int,default=0);p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--eval-clips',type=int,default=64);p.add_argument('--sample-steps',type=int,default=8);p.add_argument('--log-every',type=int,default=100);p.add_argument('--context-noise',type=float,default=.1);p.add_argument('--conditioning',choices=['motion','constant'],default='motion');p.add_argument('--resume')
    Pilot(p.parse_args()).train()
if __name__=='__main__':main()
