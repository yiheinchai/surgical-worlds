from pathlib import Path
import sys
sys.path.insert(0,str(Path('work/picodoom').resolve()))
import cv2,h5py,numpy as np,json
from models.residual_events import infer_events
m=np.load('work/readiness_checkpoint/motion/motion_codes.npz');e=np.load('work/residual_events_k4.npz');bank={'flow_centers':m['centers'],'event_centers':e['centers'],'event_pca_mean':e['pca_mean'],'event_pca_components':e['pca_components']};starts=np.linspace(47000,52997,256,dtype=int)
with h5py.File('work/picodoom/data/picodoom_frames.h5') as f:before=f['frames'][starts];after=f['frames'][starts+2]
mi=m['ids'][starts];ids=infer_events(before,after,mi,bank);proto=(e['pca_mean']+e['centers']@e['pca_components']).reshape(4,8,8,3);proto=np.stack([cv2.resize(x,(24,26)) for x in proto]);yy,xx=np.mgrid[:64,:64].astype(np.float32);vals={k:[] for k in ['warp','inferred','shuffled','modal']}
for j,(prev,target,motion,event) in enumerate(zip(before,after,mi,ids)):
 flow=cv2.resize(m['centers'][motion].reshape(4,4,2),(64,64));warped=cv2.remap(prev.astype(np.float32)/127.5-1,xx-flow[...,0],yy-flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[32:58,20:44];y=target[32:58,20:44].astype(np.float32)/127.5-1
 for name,p in [('warp',warped),('inferred',warped+proto[event]),('shuffled',warped+proto[ids[j-1]]),('modal',warped+proto[0])]:
  p=p.clip(-1,1);vals[name].append({'l1':float(abs(p-y).mean()),'mse':float(((p-y)**2).mean())})
report={'means':{k:{metric:float(np.mean([r[metric] for r in v])) for metric in ['l1','mse']} for k,v in vals.items()},'event_counts':np.bincount(ids,minlength=4).tolist(),'source_starts':starts.tolist(),'per_clip':vals,'purpose':'Bounded information test of training-only event prototypes on reused temporal development RGB. Not neural generation or causal firing evidence.'};Path('work/event_information.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['means'],indent=2))
