"""Small RGB-conditioned EDM for a video-only world-model pilot.

EDM preconditioning follows Karras et al. (2022); history corruption is inspired
by GameNGen / diffusion-forcing world models. Implemented locally; no external
model weights or supervised action labels. Actions are fitted RGB motion codes.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,cin,cout,emb):
        super().__init__();self.n1=nn.GroupNorm(8,cin);self.c1=nn.Conv2d(cin,cout,3,padding=1);self.n2=nn.GroupNorm(8,cout);self.c2=nn.Conv2d(cout,cout,3,padding=1);self.emb=nn.Linear(emb,2*cout);self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
        nn.init.zeros_(self.c2.weight);nn.init.zeros_(self.c2.bias)
    def forward(self,x,e):
        h=self.c1(F.silu(self.n1(x)));g,b=self.emb(F.silu(e)).chunk(2,-1)
        h=self.c2(F.silu(self.n2(h)*(1+g[:,:,None,None])+b[:,:,None,None]))
        return (self.skip(x)+h)/math.sqrt(2)

class PixelDenoiser(nn.Module):
    def __init__(self,codes=8,history=4,width=32):
        super().__init__();self.codes=codes;self.history=history;self.width=width;em=width*4
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),32)))
        self.time=nn.Sequential(nn.Linear(64+history,em),nn.SiLU(),nn.Linear(em,em))
        self.actions=nn.Sequential(nn.Linear(codes*history,em),nn.SiLU(),nn.Linear(em,em))
        self.input=nn.Conv2d(3*(history+1),width,3,padding=1)
        self.b0=nn.ModuleList([Block(width,width,em) for _ in range(2)])
        self.down1=nn.Conv2d(width,width*2,3,stride=2,padding=1)
        self.b1=nn.ModuleList([Block(width*2,width*2,em) for _ in range(2)])
        self.down2=nn.Conv2d(width*2,width*4,3,stride=2,padding=1)
        self.b2=nn.ModuleList([Block(width*4,width*4,em) for _ in range(2)])
        self.mid=nn.ModuleList([Block(width*4,width*4,em) for _ in range(2)])
        self.u1=nn.ModuleList([Block(width*6,width*2,em),Block(width*2,width*2,em)])
        self.u0=nn.ModuleList([Block(width*3,width,em),Block(width,width,em)])
        self.out=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv2d(width,3,3,padding=1))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
    def forward(self,noisy,sigma,history,actions,history_noise=None):
        b=noisy.shape[0];sigma=sigma.reshape(b)
        if history_noise is None:history_noise=torch.zeros(b,self.history,device=noisy.device)
        ang=sigma.log()[:,None]/4*self.frequencies[None]
        e=self.time(torch.cat([ang.cos(),ang.sin(),history_noise],-1))+self.actions(F.one_hot(actions,self.codes).float().flatten(1))
        ci=(sigma**2+.25).rsqrt()[:,None,None,None]
        h=self.input(torch.cat([noisy*ci,history.flatten(1,2)*2],1))
        for m in self.b0:h=m(h,e)
        s0=h;h=self.down1(h)
        for m in self.b1:h=m(h,e)
        s1=h;h=self.down2(h)
        for m in self.b2:h=m(h,e)
        for m in self.mid:h=m(h,e)
        h=torch.cat([F.interpolate(h,size=s1.shape[-2:],mode='nearest'),s1],1)
        for m in self.u1:h=m(h,e)
        h=torch.cat([F.interpolate(h,size=s0.shape[-2:],mode='nearest'),s0],1)
        for m in self.u0:h=m(h,e)
        raw=self.out(h);skip=(.25/(sigma**2+.25))[:,None,None,None];co=(sigma*.5/(sigma**2+.25).sqrt())[:,None,None,None]
        return skip*noisy+co*raw

    @torch.no_grad()
    def sample(self,history,actions,steps=8,seed=0,stabilization=0.,heun=True):
        if steps<2:raise ValueError('At least two denoising steps required')
        device=history.device;b=history.shape[0];g=torch.Generator(device=device).manual_seed(seed)
        schedule=torch.linspace(0,1,steps,device=device)
        sigmas=(5**(1/7)+schedule*(.002**(1/7)-5**(1/7)))**7
        sigmas=F.pad(sigmas,(0,1));x=torch.randn(history[:,-1].shape,device=device,generator=g)*sigmas[0]
        hn=torch.full((b,self.history),stabilization,device=device)
        # Reuse the same context/noise across solver steps.
        hist=history+torch.randn(history.shape,device=device,generator=g)*stabilization if stabilization else history
        for i in range(steps):
            s,sn=sigmas[i:i+2];d=self(x,s.expand(b),hist,actions,hn).clamp(-1,1);der=(x-d)/s;xnext=x+(sn-s)*der
            if heun and sn>0:
                dn=self(xnext,sn.expand(b),hist,actions,hn).clamp(-1,1);xnext=x+(sn-s)*(der+(xnext-dn)/sn)/2
            x=xnext
        return x.clamp(-1,1)

def denoising_loss(model,history,target,actions,context_noise=.1):
    b=len(target);s=(torch.randn(b,device=target.device)*1.2-1.2).exp().clamp(.002,5)
    hn=torch.rand(b,model.history,device=target.device)*context_noise
    hist=history+torch.randn_like(history)*hn[:,:,None,None,None]
    noise=torch.randn_like(target);noisy=target+noise*s[:,None,None,None]
    pred=model(noisy,s,hist,actions,hn)
    weights=((s**2+.25)/(s*.5)**2)[:,None,None,None]
    per=((pred-target)**2*weights).mean((1,2,3))
    return per.mean(),{'sigma_mean':float(s.mean()),'context_noise_mean':float(hn.mean()),'denoising_l1':float((pred.detach()-target).abs().mean())}
