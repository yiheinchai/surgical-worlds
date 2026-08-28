from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import argparse
import os
from omegaconf import OmegaConf

from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy
import torch


class DeviceType(str, Enum):
	CUDA: str = 'cuda'
	CPU: str = 'cpu'

@dataclass
class FSDPMixedPrecisionConfig:
    param_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    output_dtype: str = "float32"
    cast_forward_inputs: bool = True

    def _resolve_dtype(self, value: str | torch.dtype) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        try:
            return getattr(torch, value)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype '{value}'") from exc

    def to_policy(self) -> MixedPrecisionPolicy:
        return MixedPrecisionPolicy(
            param_dtype=self._resolve_dtype(self.param_dtype),
            reduce_dtype=self._resolve_dtype(self.reduce_dtype),
            output_dtype=self._resolve_dtype(self.output_dtype),
            cast_forward_inputs=self.cast_forward_inputs,
        )


@dataclass
class DistributedConfig:
	use_ddp: bool = False
	use_fsdp: bool = False
	reshard_after_forward: bool = False
	fsdp_mixed_precision: FSDPMixedPrecisionConfig | None = field(default_factory=FSDPMixedPrecisionConfig)
	offload_policy: CPUOffloadPolicy | None = None

	def __post_init__(self) -> None:
		if self.use_ddp and self.use_fsdp:
			raise ValueError("DistributedConfig cannot enable both DDP and FSDP; choose only one.")

	def get_mixed_precision_policy(self) -> MixedPrecisionPolicy | None:
		if self.fsdp_mixed_precision is None:
			return None
		if isinstance(self.fsdp_mixed_precision, MixedPrecisionPolicy):
			return self.fsdp_mixed_precision
		return self.fsdp_mixed_precision.to_policy()


def _validate_amp_fsdp(amp: bool, distributed: DistributedConfig) -> None:
	if amp and distributed.use_fsdp:
		raise ValueError(
			"Disable AMP when using FSDP; configure mixed precision via distributed.fsdp_mixed_precision instead."
		)


def _validate_distibuted_training(nproc_per_node: int, distributed: DistributedConfig) -> None:
	if nproc_per_node > 1 and not (distributed.use_ddp or distributed.use_fsdp):
		raise ValueError(
			"nproc_per_node > 1 requires enabling distributed.use_ddp or distributed.use_fsdp."
		)


def _validate_distributed_device(device: DeviceType, distributed: DistributedConfig) -> None:
	current_device = device if isinstance(device, DeviceType) else DeviceType(device)
	if (distributed.use_ddp or distributed.use_fsdp) and current_device is not DeviceType.CUDA:
		raise ValueError("Distributed training (DDP/FSDP) requires device=cuda.")


@dataclass
class VideoTokenizerConfig:
	batch_size_per_gpu: int
	gradient_accumulation_steps: int
	n_updates: int
	learning_rate: float
	log_interval: int
	dataset: str
	context_length: int
	frame_size: int
	patch_size: int
	embed_dim: int
	num_heads: int
	hidden_dim: int
	num_blocks: int
	latent_dim: int
	num_bins: int
	amp: bool
	tf32: bool
	compile: bool
	distributed: DistributedConfig
	nproc_per_node: int
	standalone: bool
	use_wandb: bool
	wandb_project: str
	checkpoint: Optional[str]
	optimizer: str = "adamw"
	muon_momentum: float = 0.95
	muon_backend_steps: int = 5
	device: DeviceType = DeviceType.CUDA
	fps: Optional[int] = None
	preload_ratio: Optional[float] = None
	
	def __post_init__(self) -> None:
		_validate_amp_fsdp(self.amp, self.distributed)
		_validate_distibuted_training(self.nproc_per_node, self.distributed)
		_validate_distributed_device(self.device, self.distributed)


@dataclass
class LatentActionsConfig:
	batch_size_per_gpu: int
	gradient_accumulation_steps: int
	n_updates: int
	learning_rate: float
	log_interval: int
	dataset: str
	context_length: int
	frame_size: int
	n_actions: int
	patch_size: int
	embed_dim: int
	num_heads: int
	hidden_dim: int
	num_blocks: int
	amp: bool
	tf32: bool
	compile: bool
	distributed: DistributedConfig
	nproc_per_node: int
	standalone: bool
	use_wandb: bool
	wandb_project: str
	checkpoint: Optional[str]
	optimizer: str = "adamw"
	muon_momentum: float = 0.95
	muon_backend_steps: int = 5
	device: DeviceType = DeviceType.CUDA
	fps: Optional[int] = None
	preload_ratio: Optional[float] = None
	
	def __post_init__(self) -> None:
		_validate_amp_fsdp(self.amp, self.distributed)
		_validate_distibuted_training(self.nproc_per_node, self.distributed)
		_validate_distributed_device(self.device, self.distributed)


@dataclass
class DynamicsConfig:
	batch_size_per_gpu: int
	gradient_accumulation_steps: int
	n_updates: int
	learning_rate: float
	log_interval: int
	dataset: str
	context_length: int
	frame_size: int
	patch_size: int
	embed_dim: int
	num_heads: int
	hidden_dim: int
	num_blocks: int
	latent_dim: int
	num_bins: int
	n_actions: int
	use_actions: bool
	video_tokenizer_path: Optional[str]
	latent_actions_path: Optional[str]
	amp: bool
	tf32: bool
	compile: bool
	distributed: DistributedConfig
	nproc_per_node: int
	standalone: bool
	use_wandb: bool
	wandb_project: str
	checkpoint: Optional[str]
	use_moe: bool = False
	num_experts: int = 4
	top_k_experts: int = 2
	moe_aux_loss_coeff: float = 0.01
	optimizer: str = "adamw"
	muon_momentum: float = 0.95
	muon_backend_steps: int = 5
	device: DeviceType = DeviceType.CUDA
	fps: Optional[int] = None
	preload_ratio: Optional[float] = None
	
	def __post_init__(self) -> None:
		_validate_amp_fsdp(self.amp, self.distributed)
		_validate_distibuted_training(self.nproc_per_node, self.distributed)
		_validate_distributed_device(self.device, self.distributed)


@dataclass
class TrainingConfig:
	use_wandb: bool
	wandb_project: str
	dataset: str
	video_tokenizer_config: str
	latent_actions_config: str
	dynamics_config: str
	run_video_tokenizer: bool
	run_latent_actions: bool
	run_dynamics: bool
	patch_size: int
	context_length: int
	frame_size: int
	latent_dim: int
	num_bins: int
	n_actions: int
	amp: bool
	tf32: bool
	compile: bool
	distributed: DistributedConfig
	nproc_per_node: int
	standalone: bool
	device: DeviceType = DeviceType.CUDA
	embed_dim: Optional[int] = None
	num_heads: Optional[int] = None
	hidden_dim: Optional[int] = None
	num_blocks: Optional[int] = None
	learning_rate: Optional[float] = None
	batch_size_per_gpu: Optional[int] = None
	gradient_accumulation_steps: Optional[int] = None
	log_interval: Optional[int] = None
	n_updates: Optional[int] = None
	fps: Optional[int] = None
	preload_ratio: Optional[float] = None
	use_moe: bool = False
	num_experts: int = 4
	top_k_experts: int = 2
	moe_aux_loss_coeff: float = 0.01
	optimizer: str = "adamw"
	muon_momentum: float = 0.95
	muon_backend_steps: int = 5
	
	def __post_init__(self) -> None:
		_validate_amp_fsdp(self.amp, self.distributed)
		_validate_distibuted_training(self.nproc_per_node, self.distributed)
		_validate_distributed_device(self.device, self.distributed)


@dataclass
class InferenceConfig:
	video_tokenizer_path: Optional[str]
	latent_actions_path: Optional[str]
	dynamics_path: Optional[str]
	device: str
	generation_steps: int
	context_window: int
	fps: int
	temperature: float
	use_actions: bool
	teacher_forced: bool
	use_latest_checkpoints: bool
	prediction_horizon: int
	dataset: str
	use_gt_actions: bool
	amp: bool
	tf32: bool
	compile: bool
	use_interactive_mode: bool
	preload_ratio: Optional[float] = None


def load_config(config_cls, default_config_path: Optional[str] = None):
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument("--config", type=str, default=default_config_path)
	parser.add_argument("overrides", nargs=argparse.REMAINDER)
	args = parser.parse_args()
	base = OmegaConf.structured(config_cls)
	cfg = base
	if args.config is not None:
		if not os.path.isfile(args.config):
			raise FileNotFoundError(f"Config file not found: {args.config} (cwd: {os.getcwd()})")
		file_cfg = OmegaConf.load(args.config)
		cfg = OmegaConf.merge(cfg, file_cfg)
	dot_overrides = [s.lstrip('-') for s in (args.overrides or []) if '=' in s]
	if dot_overrides:
		cli_cfg = OmegaConf.from_dotlist(dot_overrides)
		cfg = OmegaConf.merge(cfg, cli_cfg)
	return OmegaConf.to_object(cfg)


def load_stage_config_merged(config_cls, default_config_path: Optional[str] = None, training_config_path: Optional[str] = None):
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument("--config", type=str, default=default_config_path)
	parser.add_argument("--training_config", type=str, default=training_config_path or os.path.join(os.getcwd(), 'configs', 'training.yaml'))
	parser.add_argument("overrides", nargs=argparse.REMAINDER)
	args = parser.parse_args()
	base = OmegaConf.structured(config_cls)
	cfg = base
	if args.config is not None:
		if not os.path.isfile(args.config):
			raise FileNotFoundError(f"Config file not found: {args.config} (cwd: {os.getcwd()})")
		stage_file_cfg = OmegaConf.load(args.config)
		cfg = OmegaConf.merge(cfg, stage_file_cfg)
	if args.training_config and os.path.isfile(args.training_config):
		training_file_cfg = OmegaConf.load(args.training_config)
		training_container = OmegaConf.to_container(training_file_cfg, resolve=True) or {}
		allowed_keys = set(cfg.keys())
		filtered_training = {k: v for k, v in training_container.items() if k in allowed_keys and v is not None}
		if filtered_training:
			training_cfg_filtered = OmegaConf.create(filtered_training)
			cfg = OmegaConf.merge(cfg, training_cfg_filtered)
	dot_overrides = [s.lstrip('-') for s in (args.overrides or []) if '=' in s]
	if dot_overrides:
		cli_cfg = OmegaConf.from_dotlist(dot_overrides)
		cfg = OmegaConf.merge(cfg, cli_cfg)
	return OmegaConf.to_object(cfg)
