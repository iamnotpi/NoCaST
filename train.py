import os
import time

import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers import AutoencoderKL

from model import NCST, NCSTConfig
from score_matching import denoising_score_matching
from langevin_dynamics import annealed_Langevin_dynamics

config = NCSTConfig(
    dim=768,
    n_heads=12,
    n_blocks=12,
    total_steps=10,
    patch_size=2,
    noise_dim=(4, 16, 16)
)

data_path = 'afhq_v2/train'
image_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bsz = 128
sigmas = torch.logspace(0, -2, 10, base=10, device=device)

def center_crop_arr(pil_image, image_size):
    """
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

transform = transforms.Compose([
    transforms.Lambda(lambda x: center_crop_arr(x, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = ImageFolder(data_path, transform=transform)

train_loader = DataLoader(
    dataset,
    batch_size=bsz,
    shuffle=True,
    num_workers=4,  
    pin_memory=True,
    drop_last=True
)

torch.set_float32_matmul_precision('high')

# The author used the ema variant for training DiT
vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-ema').to(device)

ckpt = torch.load('logs/models/checkpoint_55000.pt')

model = NCST(config).to(device)
model = torch.compile(model)
model.load_state_dict(ckpt['model_state_dict'])

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Using device: {device}")

use_fused = (device == 'cuda')
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, fused=use_fused)
optimizer.load_state_dict(ckpt['optimizer_state_dict'])

train_steps = ckpt['train_steps'] + 1
# train_steps = 0
total_train_steps = 65000

log_dir = 'logs'
models_dir = os.path.join(log_dir, 'models')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
writer = SummaryWriter(f'{log_dir}/tensorboard')

while train_steps < total_train_steps:
    for i, (X, _) in enumerate(train_loader):
        if train_steps >= total_train_steps:
            break
        
        start = time.time()
        X = X.to(device)
        noise_level = torch.randint(0, config.total_steps, (X.shape[0],), device=device)

        with torch.no_grad():
            X = vae.encode(X).latent_dist.sample().mul_(0.18215)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            loss = denoising_score_matching(model, X, noise_level, sigmas, device=device)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (train_steps + 1) % 1000 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_steps': train_steps
            }
            checkpoint_path = os.path.join(models_dir, f'checkpoint_{train_steps + 1}.pt')
            torch.save(checkpoint, checkpoint_path)
            # Generate samples
            with torch.no_grad():
                # Generate latents using existing function
                latents = annealed_Langevin_dynamics(
                    scorenet=model,
                    num_imgs=4,
                    noise_dim=config.noise_dim,
                    sigmas=sigmas,
                    device=device
                )
                
                latents = 1 / 0.18215 * latents
                images = vae.decode(latents).sample
                images = (images / 2 + 0.5).clamp(0, 1)
                
                # Create and log image grid
                grid = make_grid(images, nrow=2, normalize=False)
                writer.add_image('samples', grid, train_steps + 1)
        
        torch.cuda.synchronize()
        end = time.time()

        step_time = (end - start) * 1000
        writer.add_scalar('training/loss', loss.item(), train_steps + 1)
        writer.add_scalar('training/norm', norm.item(), train_steps + 1)
        writer.add_scalar('training/step_time_ms', step_time, train_steps + 1)

        train_steps += 1

writer.close()
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_steps': train_steps
}
checkpoint_path = os.path.join(models_dir, f'checkpoint_last.pt')
torch.save(checkpoint, checkpoint_path)