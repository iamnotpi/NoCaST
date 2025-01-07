import os
import time

import torch
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from diffusers import AutoencoderKL

from unet import NCSN, UNetConfig

def annealed_Langevin_dynamics(scorenet, num_imgs, noise_dim, sigmas, T=100, eps=2e-5, device='cuda'):
    scorenet.eval()
    x = torch.rand(num_imgs, *noise_dim, device=device)
    with torch.no_grad():
        for i, sigma in enumerate(sigmas):
            alpha = eps * (sigma / sigmas[-1]) ** 2 
            for _ in range(T):
                z = torch.randn_like(x)
                grad = scorenet(x, torch.tensor([i] * num_imgs, device=device))
                x = x + (alpha / 2) * grad + alpha ** 0.5 * z
        return x

def denoising_score_matching(scorenet, X, noise_level, sigmas, device):
    used_sigmas = sigmas[noise_level].view(X.shape[0], *([1] * len(X.shape[1:]))) # (B, 1, ..., 1)
    X_tilde = X + torch.randn_like(X, device=device) * used_sigmas
    scores = scorenet(X_tilde, noise_level)
    target = - 1 / (used_sigmas ** 2) * (X_tilde - X)
    target = target.reshape(target.shape[0], -1)
    scores = scores.reshape(scores.shape[0], -1)
    loss = 0.5 * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** 2
    return loss.mean(0)

class PrepDataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.data = torch.from_numpy(np.load(data_path)).type(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

data_path = 'afhq_v2/train'
image_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bsz = 128
sigmas = torch.logspace(0, -2, 10, base=10, device=device)

dataset = PrepDataset('all_features.npy')

train_loader = DataLoader(
    dataset,
    batch_size=bsz,
    shuffle=True,
    num_workers=4,  
    pin_memory=True,
    drop_last=True
)

torch.set_float32_matmul_precision('high')

# Same as DiT
vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-ema').to(device)

config = UNetConfig()
model = NCSN(config).to(device)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Using device: {device}")

use_fused = (device == 'cuda')
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, fused=use_fused)

train_steps = 0
total_train_steps = 283000

log_dir = 'unet_logs'
models_dir = os.path.join(log_dir, 'models')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
writer = SummaryWriter(f'{log_dir}/tensorboard')

data = torch.from_numpy(np.load('all_features.npy')).type(torch.float32).to(device)
bsz = 128
while train_steps < total_train_steps:
    
    idx = torch.randint(0, data.size(0), (bsz,))
    X = data[idx]
    start = time.time()
    noise_level = torch.randint(0, config.total_steps, (X.shape[0],), device=device)

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
            latents = annealed_Langevin_dynamics(
                scorenet=model,
                num_imgs=4,
                noise_dim=(4, 16, 16),
                sigmas=sigmas,
                device=device
            )
            
            latents *= 1 / 0.18215 
            images = vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            
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