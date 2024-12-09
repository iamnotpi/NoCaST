import torch

def denoising_score_matching(scorenet, X, noise_level, sigmas, device):
    used_sigmas = sigmas[noise_level].view(X.shape[0], *([1] * len(X.shape[1:]))) # (B, 1, ..., 1)
    X_tilde = X + torch.randn_like(X, device=device) * used_sigmas
    scores = scorenet(X_tilde, noise_level)
    target = - 1 / (used_sigmas ** 2) * (X_tilde - X)
    target = target.reshape(target.shape[0], -1)
    scores = scores.reshape(scores.shape[0], -1)
    loss = 0.5 * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** 2
    return loss.mean(0)