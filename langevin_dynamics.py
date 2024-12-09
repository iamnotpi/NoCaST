import torch

def annealed_Langevin_dynamics(scorenet, num_imgs, noise_dim, sigmas, T=100, eps=2e-5, device='cuda'):
    scorenet.eval()
    x = torch.rand(num_imgs, *noise_dim, device=device)
    with torch.no_grad():
        for i, sigma in enumerate(sigmas):
            alpha = eps * (sigma / sigmas[-1]) ** 2 
            for _ in range(T):
                z = torch.randn_like(x)
                grad = scorenet(x, torch.tensor([i], device=device))
                x = x + (alpha / 2) * grad + alpha ** 0.5 * z
        return x