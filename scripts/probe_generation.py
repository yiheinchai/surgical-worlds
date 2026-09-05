"""Deterministic decoding/rollout comparison with no optimizer updates.

Video-inferred future codes are oracle diagnostic inputs, never playable
controls. No future pixels or tokenizer latents enter generated history.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from PIL import Image
import torch
from models.latent_actions import LatentActionModel
from models.video_tokenizer import VideoTokenizer
from models.dynamics import DynamicsModel
from scripts.bounded_picodoom import source_starts, per_image


@torch.no_grad()
def generate(vt, dyn, history, actions, iterations):
    if iterations == 1:
        masked = torch.cat([history, dyn.mask_token.expand(len(history), 1, history.shape[2], -1)], 1)
        logits, _, _ = dyn(masked, training=False, conditioning=actions)
        nxt = vt.quantizer.get_latents_from_indices(logits[:, -1:].argmax(-1))
        latents = torch.cat([history, nxt], 1)
    else:
        latents = dyn.forward_inference(history, 1, iterations,
            vt.quantizer.get_latents_from_indices, conditioning=actions, temperature=0.)
    return latents, vt.decoder(latents)[:, -1]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--name', default='generation_validation')
    p.add_argument('--key-file', required=True)
    p.add_argument('--data', default='data/picodoom_frames.h5')
    p.add_argument('--checkpoints', default='data/checkpoints')
    a = p.parse_args()
    torch.manual_seed(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ['WANDB_API_KEY'] = Path(a.key_file).read_text().strip()
    import wandb
    out = Path('results/bounded-20260905') / a.name; out.mkdir(parents=True, exist_ok=True)
    starts = source_starts(47000, 53000, 4, 2, 128, 701)
    rollout_starts = np.array([47000, 48500, 50000, 51500])
    config = {'models': a.models, 'validation_starts': starts.tolist(),
              'rollout_starts': rollout_starts.tolist(), 'rollout_horizon': 64,
              'iterations': [1, 16], 'temperature': 0, 'supervision': 'RGB only',
              'future_codes': 'video-inferred oracle diagnostic controls',
              'timing': 'batch 1 GPU rollout compute, excludes UI/network/I/O',
              'recache': 'reencode last three rendered RGB frames before each prediction'}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    run = wandb.init(entity='data2yihein-d', project='tinyworlds', group='picodoom-bounded-20260905',
                     name=a.name, config=config, dir=str(out))
    common = dict(frame_size=(64,64), patch_size=4, embed_dim=32, num_heads=8, hidden_dim=128)
    vt = VideoTokenizer(**common, num_blocks=4, latent_dim=5, num_bins=4).cuda().eval()
    lam = LatentActionModel(**common, num_blocks=2, n_actions=4).cuda().eval()
    dyn = DynamicsModel(**common, num_blocks=8, latent_dim=5, num_bins=4, conditioning_dim=2).cuda().eval()
    for model, stage, step in [(vt,'video_tokenizer',37500),(lam,'latent_actions',9500)]:
        path = next(Path(a.checkpoints).glob(f'**/{stage}/checkpoints/{stage}_step_{step}/model_state_dict.pt'))
        model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
    with h5py.File(a.data) as f: data = torch.from_numpy(f['frames'][:])
    def clips(indices, length):
        rows = torch.as_tensor(indices[:,None] + 2*np.arange(length)[None,:])
        return data[rows].cuda().float().permute(0,1,4,2,3)/127.5-1
    result = {}
    for entry in a.models:
        name, path = entry.split('=',1)
        if path == 'original':
            path = str(next(Path(a.checkpoints).glob('**/dynamics/checkpoints/dynamics_step_44000/model_state_dict.pt')))
        dyn.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
        metrics = {}
        for iterations in (1,16):
            values = {}
            for i in range(0,len(starts),16):
                x = clips(starts[i:i+16],4)
                lat = vt.quantizer.get_latents_from_indices(vt.tokenize(x[:,:3]))
                _, pred = generate(vt,dyn,lat,lam.encode(x),iterations)
                for key, value in per_image(pred,x[:,-1],x[:,-2]).items():
                    values.setdefault(key,[]).append(value)
            metrics.update({f'one_step/{iterations}_iterations/{k}':float(torch.cat(v).mean()) for k,v in values.items()})
            for recache in (False,True):
                all_predictions=[]; timing=[]
                for start in rollout_starts:
                    x=clips(np.array([start]),67)
                    frames=x[:,:3].clone()
                    lat=vt.quantizer.get_latents_from_indices(vt.tokenize(frames))
                    actions=[lam.encode(x[:,j-3:j+1])[:,-1] for j in range(3,67)]
                    past=lam.encode(frames);predictions=[]
                    for j in range(64):
                        past=torch.cat([past,actions[j][:,None]],1)[:,-3:]
                        torch.cuda.synchronize();t=time.perf_counter()
                        if recache:
                            lat=vt.quantizer.get_latents_from_indices(vt.tokenize(frames[:,-3:]))
                        lat,pred=generate(vt,dyn,lat[:,-3:],past,iterations)
                        torch.cuda.synchronize()
                        if j>=4:timing.append(time.perf_counter()-t)
                        frames=torch.cat([frames[:,-3:],pred[:,None]],1)
                        predictions.append(pred)
                    all_predictions.append(torch.stack(predictions,1))
                pred=torch.cat(all_predictions);truth=clips(rollout_starts,67)
                key=f'rollout/{iterations}_iterations/recache_{recache}'
                for h in (1,4,8,16,32,64):
                    metrics[f'{key}/l1_at_{h}']=float((pred[:,h-1]-truth[:,h+2]).abs().mean())
                    metrics[f'copy/l1_at_{h}']=float((truth[:,2]-truth[:,h+2]).abs().mean())
                metrics[f'{key}/batch1_ms_median']=float(np.median(timing)*1000)
                metrics[f'{key}/batch1_ms_p95']=float(np.percentile(timing,95)*1000)
                panels=torch.cat([truth[:,3:],truth[:,2:3].expand(-1,64,-1,-1,-1),pred],-1)
                frames=torch.cat(list(panels),-2).clamp(-1,1).cpu()
                imgs=[Image.fromarray(((f+1)*127.5).byte().permute(1,2,0).numpy()) for f in frames]
                gif=out/f'{name}_iterations{iterations}_recache{recache}.gif'
                imgs[0].save(gif,save_all=True,append_images=imgs[1:],duration=100,loop=0)
                run.log({f'{name}/{key}/video':wandb.Video(str(gif),caption='target | seed-frame copy | video-oracle-code rollout')})
        result[name]=metrics
        (out/'metrics.json').write_text(json.dumps(result,indent=2))
        run.log({f'{name}/{k}':v for k,v in metrics.items()})
        print(json.dumps({name:metrics}),flush=True)
    (out/'result.json').write_text(json.dumps({'status':'completed_evaluation','models_completed':list(result)},indent=2))
    artifact=wandb.Artifact(a.name,type='video-only-generation-probe')
    for f in out.iterdir():
        if f.is_file():artifact.add_file(str(f))
    run.log_artifact(artifact);run.finish()


if __name__ == '__main__':main()
