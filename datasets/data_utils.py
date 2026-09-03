import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.datasets import PongDataset, SonicDataset, PolePositionDataset, PicoDoomDataset, ZeldaDataset
from datasets.surgical_datasets import LaparoscopicDataset, RoboticLaparoscopicDataset

DEFAULT_NUM_WORKERS = 2
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_PIN_MEMORY = False
DEFAULT_PERSISTENT_WORKERS = True


def _default_video_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def _load_video_dataset_pair(dataset_cls, video_rel_path, h5_rel_path, num_frames, transform=None, fps=30, preload_ratio=1, **kwargs):
    current_folder_path = os.getcwd()
    video_path = current_folder_path + video_rel_path
    preprocessed_path = current_folder_path + h5_rel_path
    transform = _default_video_transform() if transform is None else transform

    train = dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=True,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        **kwargs
    )
    val = dataset_cls(
        video_path,
        transform=transform,
        save_path=preprocessed_path,
        train=False,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        **kwargs
    )
    return train, val


def load_pong(num_frames=1, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        PongDataset,
        '/data/pong.mp4',
        '/data/pong_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_sonic(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        SonicDataset,
        '/data/sonic_frames.mp4',
        '/data/sonic_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_pole_position(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        PolePositionDataset,
        '/data/pole_position.mp4',
        '/data/pole_position_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def load_picodoom(num_frames=4, fps=30, preload_ratio=1):
    return _load_video_dataset_pair(
        PicoDoomDataset,
        '/data/picodoom cleaned.mp4',
        '/data/picodoom_frames.h5',
        num_frames=num_frames,
        fps=30,
        preload_ratio=preload_ratio
    )


def load_zelda(num_frames=4, fps=15, preload_ratio=1):
    return _load_video_dataset_pair(
        ZeldaDataset,
        '/data/Zelda oot2d 1 Cut.mp4',
        '/data/zelda_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio
    )


def _load_surgical_dataset_pair(
    dataset_cls,
    video_rel_path: str,
    train_h5_rel_path: str,
    val_h5_rel_path: str,
    num_frames: int,
    transform=None,
    fps=10,
    preload_ratio=1,
    **kwargs
):
    current_folder_path = os.getcwd()
    video_path = current_folder_path + video_rel_path
    train_h5 = current_folder_path + train_h5_rel_path
    val_h5 = current_folder_path + val_h5_rel_path
    transform = _default_video_transform() if transform is None else transform
    manifest_path = os.path.join(os.path.dirname(train_h5), "manifest.json")

    train = dataset_cls(
        video_path,
        transform=transform,
        save_path=train_h5 if os.path.exists(train_h5) else None,
        train=True,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        manifest_path=manifest_path if os.path.exists(manifest_path) else None,
        **kwargs,
    )
    val = dataset_cls(
        video_path,
        transform=transform,
        save_path=val_h5 if os.path.exists(val_h5) else train_h5,
        train=False,
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        manifest_path=manifest_path if os.path.exists(manifest_path) else None,
        **kwargs,
    )
    return train, val


def load_laparoscopic(num_frames=4, fps=10, preload_ratio=1, resolution=(128, 128)):
    return _load_surgical_dataset_pair(
        LaparoscopicDataset,
        '/data/surgical/laparoscopic/videos',
        '/data/surgical/train_laparoscopic_frames.h5',
        '/data/surgical/val_laparoscopic_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        resolution=resolution,
    )


def load_robotic_laparoscopic(num_frames=4, fps=10, preload_ratio=1, resolution=(128, 128)):
    return _load_surgical_dataset_pair(
        RoboticLaparoscopicDataset,
        '/data/surgical/robotic/videos',
        '/data/surgical/train_robotic_frames.h5',
        '/data/surgical/val_robotic_frames.h5',
        num_frames=num_frames,
        fps=fps,
        preload_ratio=preload_ratio,
        resolution=resolution,
    )


def data_loaders(train_data, val_data, batch_size, distributed=False, rank=0, world_size=1):
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False if train_sampler is not None else True,
        sampler=train_sampler,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=DEFAULT_PERSISTENT_WORKERS,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False if val_sampler is not None else True,
        sampler=val_sampler,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=DEFAULT_PERSISTENT_WORKERS,
        prefetch_factor=DEFAULT_PREFETCH_FACTOR,
        drop_last=True
    )
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, num_frames=1, distributed=False, rank=0, world_size=1, fps=15, preload_ratio=1):
    if dataset == 'PONG':
        training_data, validation_data = load_pong(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'SONIC':
        training_data, validation_data = load_sonic(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'POLE_POSITION':
        training_data, validation_data = load_pole_position(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'PICODOOM':
        training_data, validation_data = load_picodoom(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'ZELDA':
        training_data, validation_data = load_zelda(num_frames=num_frames, fps=fps, preload_ratio=preload_ratio)
    elif dataset == 'LAPAROSCOPIC':
        training_data, validation_data = load_laparoscopic(
            num_frames=num_frames, fps=fps, preload_ratio=preload_ratio
        )
    elif dataset == 'ROBOTIC_LAPAROSCOPIC':
        training_data, validation_data = load_robotic_laparoscopic(
            num_frames=num_frames, fps=fps, preload_ratio=preload_ratio
        )
    else:
        raise ValueError(
            f'Invalid dataset: {dataset}. '
            'Supported: PONG, SONIC, POLE_POSITION, PICODOOM, ZELDA, LAPAROSCOPIC, ROBOTIC_LAPAROSCOPIC'
        )

    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size,
        distributed=distributed, rank=rank, world_size=world_size
    )
    x_train_var = np.var(training_data.data)

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def visualize_reconstruction(original, reconstruction, save_path=None):
    original = original.detach().to('cpu', dtype=torch.float32)
    reconstruction = reconstruction.detach().to('cpu', dtype=torch.float32)

    if original.dim() == 4:
        original = original.unsqueeze(1)
    if reconstruction.dim() == 4:
        reconstruction = reconstruction.unsqueeze(1)

    num_sequences = min(4, original.shape[0])
    seq_length = min(4, original.shape[1])

    original = original[:num_sequences, :seq_length]
    reconstruction = reconstruction[:num_sequences, :seq_length]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    orig_flat = original.reshape(-1, *original.shape[2:])
    # Use value_range=(-1,1) so [-1,1] tensors are mapped directly to [0,1] — collapse is visible
    grid_orig = make_grid(orig_flat, nrow=seq_length, normalize=True, value_range=(-1, 1), padding=2).clamp(0, 1)
    ax1.imshow(grid_orig.permute(1, 2, 0).contiguous().numpy())
    ax1.axis('off')
    ax1.set_title(f'Original Sequences (4 sequences x {seq_length} frames)')

    recon_flat = reconstruction.reshape(-1, *reconstruction.shape[2:])
    grid_recon = make_grid(recon_flat, nrow=seq_length, normalize=True, value_range=(-1, 1), padding=2).clamp(0, 1)
    ax2.imshow(grid_recon.permute(1, 2, 0).contiguous().numpy())
    ax2.axis('off')
    ax2.set_title(f'Reconstructed Sequences (4 sequences x {seq_length} frames)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
