"""Classical RGB flow plus train-only clustering: an unsupervised motion bottleneck.

No pretrained flow weights, engine actions, text or state labels are consumed.
Flow can describe exogenous changes and is not a measurement of player intent.
"""
import cv2
import numpy as np

def rgb_flow(a,b):
    if a.dtype!=np.uint8 or a.shape!=b.shape: raise ValueError('Matching uint8 HWC RGB required')
    return cv2.calcOpticalFlowFarneback(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),None,.5,3,15,5,7,1.5,0)

def flow_descriptor(f,grid=4):
    h,w,c=f.shape
    assert c==2 and h%grid==0 and w%grid==0
    tiles=f.reshape(grid,h//grid,grid,w//grid,2).transpose(0,2,1,3,4)
    return np.median(tiles.reshape(grid,grid,-1,2),axis=2).reshape(-1).astype(np.float32)

def distances(x,c):return ((x[:,None]-c[None])**2).mean(-1)
def assign_codes(x,c):return distances(np.asarray(x),c).argmin(1)

def fit_codebook(x,codes=8,seed=0,iterations=100,restarts=8):
    """Deterministic K-means++, fitted on training RGB-derived descriptors only."""
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or len(x)<codes or not np.isfinite(x).all():raise ValueError('Invalid training descriptors')
    rng=np.random.default_rng(seed);best=None
    for _ in range(restarts):
        c=[x[rng.integers(len(x))]]
        for _ in range(codes-1):
            d=distances(x,np.array(c)).min(1).astype(np.float64)
            i=rng.choice(len(x),p=d/d.sum()) if d.sum()>0 else rng.integers(len(x))
            c.append(x[i])
        c=np.array(c)
        for _ in range(iterations):
            d=distances(x,c);ids=d.argmin(1)
            new=np.stack([x[ids==k].mean(0) if (ids==k).any() else x[d.min(1).argmax()] for k in range(codes)])
            done=np.allclose(new,c,atol=1e-5);c=new
            if done:break
        err=float(distances(x,c).min(1).mean())
        if best is None or err<best[0]:best=(err,c.copy())
    return best[1]

def code_metrics(ids,codes):
    p=np.bincount(ids,minlength=codes)/len(ids);h=float(-(p*np.log(p.clip(1e-12))).sum())
    return dict(entropy_nats=h,effective_codes=float(np.exp(h)),fractions=p.tolist())
