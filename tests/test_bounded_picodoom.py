import numpy as np
import pytest
from pathlib import Path
import torch

from scripts.bounded_picodoom import source_starts, per_image, code_stats, soft_code_information
from models.latent_actions import LatentActionModel
from scripts.probe_action_semantics import uniform_color_fraction
from scripts.synthetic_branched_motion import translated_pairs
from scripts.probe_generation import generate
from models.dynamics import DynamicsModel
from models.video_tokenizer import VideoTokenizer
from scripts.probe_synthetic_codes import code_conditional_entropy
from utils.config import InferenceConfig, load_config


def test_source_clips_are_disjoint_and_do_not_cross_boundaries():
    sets=[]
    for lo,hi in [(300,46000),(47000,53000),(54000,59785)]:
        starts=source_starts(lo,hi,4,2,4096,9)
        indices=(starts[:,None]+np.arange(4)[None,:]*2).flatten()
        assert indices.min()>=lo and indices.max()<hi
        assert len(set(indices))==len(indices)
        sets.append(set(indices))
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]


def test_pair_decoder_has_action_gradients_and_no_target_pixel_leakage():
    torch.manual_seed(21)
    model=LatentActionModel(frame_size=(8,8),patch_size=4,embed_dim=12,
                           num_heads=3,hidden_dim=24,num_blocks=1,n_actions=4).eval()
    x=torch.rand(3,2,3,8,8)
    a=torch.randn(3,1,2,requires_grad=True)
    out=model.decoder(x,a)
    changed=x.clone();changed[:,1]=99
    assert torch.equal(out,model.decoder(changed,a))
    out.square().mean().backward()
    assert a.grad is not None and a.grad.abs().sum()>0
    assert not torch.equal(out,model.decoder(x,-a))


def test_discrete_collapse_is_visible_even_with_encoder_variance():
    model=LatentActionModel(frame_size=(8,8),patch_size=4,embed_dim=12,
                           num_heads=3,hidden_dim=24,num_blocks=1,n_actions=4)
    z=torch.tensor([[[1.,1.]],[[2.,3.]],[[4.,2.]]])
    stats=code_stats(z,model.quantizer(z),model.quantizer)
    assert stats['encoder_variance']>.01
    assert stats['unique_codes']==1 and stats['entropy_nats']==0


def test_motion_metric_exposes_copy_error_and_information_proxy_is_finite():
    prev=torch.zeros(2,3,8,8);target=prev.clone();target[:,:,:4,:4]=1
    perfect=per_image(target,target,prev);copy=per_image(prev,target,prev)
    assert all(torch.equal(v,torch.zeros_like(v)) for v in perfect.values())
    assert torch.all(copy['motion_weighted_l1']>copy['l1'])
    z=torch.randn(8,1,2,requires_grad=True)
    loss=soft_code_information(z);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_color_probe_separates_uniform_tints_from_spatial_changes():
    pred=torch.zeros(2,4,3,8,8)
    pred[:,0]=1
    assert torch.isclose(uniform_color_fraction(pred),torch.tensor(1.))
    pred[:,0,:,::2]=1;pred[:,0,:,1::2]=-1
    assert uniform_color_fraction(pred)==0


def test_branched_rgb_probe_has_ambiguous_successors_without_color_cues():
    torch.manual_seed(7)
    textures=torch.rand(2,3,16,16)
    pairs=translated_pairs(textures).reshape(2,4,2,3,16,16)
    assert torch.equal(pairs[:,0,0],textures)
    assert torch.equal(pairs[:,0,0],pairs[:,3,0])
    assert not torch.equal(pairs[:,0,1],pairs[:,1,1])
    # Translation must not introduce a mean-brightness shortcut.
    assert torch.allclose(pairs[:,:,0].mean((-1,-2)),pairs[:,:,1].mean((-1,-2)))


def test_generation_is_deterministic_and_preserves_context():
    torch.manual_seed(18)
    common=dict(frame_size=(8,8),patch_size=4,embed_dim=12,num_heads=3,hidden_dim=24,num_blocks=1)
    vt=VideoTokenizer(**common,latent_dim=2,num_bins=2).eval()
    dyn=DynamicsModel(**common,latent_dim=2,num_bins=2,conditioning_dim=2).eval()
    history=vt.quantizer(torch.randn(2,3,4,2))
    original=history.clone();actions=torch.randn(2,3,2)
    for iterations in (1,16):
        latent,pred=generate(vt,dyn,history,actions,iterations)
        latent2,pred2=generate(vt,dyn,history,actions,iterations)
        assert torch.equal(latent,latent2) and torch.equal(pred,pred2)
        assert torch.equal(latent[:,:3],original) and torch.equal(history,original)
        assert pred.shape==(2,3,8,8) and torch.isfinite(pred).all()
        production=dyn.forward_inference(history,1,iterations,vt.quantizer.get_latents_from_indices,
            conditioning=actions,decoding_mode='one_pass' if iterations==1 else 'maskgit')
        assert torch.equal(production,latent)
    with pytest.raises(ValueError,match='prediction_horizon=1'):
        dyn.forward_inference(history,2,1,vt.quantizer.get_latents_from_indices,decoding_mode='one_pass')
    with pytest.raises(ValueError,match='Unknown decoding_mode'):
        dyn.forward_inference(history,1,1,vt.quantizer.get_latents_from_indices,decoding_mode='invalid')


def test_within_scene_code_metric_rejects_balanced_appearance_codes():
    appearance_codes=torch.arange(4)[:,None].expand(4,4)
    motion_codes=torch.arange(4)[None,:].expand(4,4)
    # Same global histogram, different dependence on the successor within a scene.
    assert torch.equal(torch.bincount(appearance_codes.flatten()),torch.bincount(motion_codes.flatten()))
    assert code_conditional_entropy(appearance_codes)==0
    assert np.isclose(code_conditional_entropy(motion_codes),np.log(4))


def test_inference_cli_preserves_legacy_defaults_and_accepts_one_pass(monkeypatch):
    config=str(Path(__file__).resolve().parents[1]/'configs/inference.yaml')
    monkeypatch.setattr('sys.argv',['run_inference.py','--config',config])
    legacy=load_config(InferenceConfig)
    assert legacy.decoding_mode=='maskgit' and legacy.maskgit_steps==10
    monkeypatch.setattr('sys.argv',['run_inference.py','--config',config,
        'decoding_mode=one_pass','prediction_horizon=1','temperature=0'])
    selected=load_config(InferenceConfig)
    assert selected.decoding_mode=='one_pass' and selected.prediction_horizon==1 and selected.temperature==0
