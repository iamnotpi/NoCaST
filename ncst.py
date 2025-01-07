from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# DiT-B
@dataclass
class NCSTConfig:
    dim: int = 768
    n_heads: int = 12
    n_blocks: int = 12
    total_steps: int = 10
    patch_size: int = 4
    noise_dim: tuple = (3, 64, 64)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, dim, patch_size, bias=True):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=bias),
            nn.Flatten(2)
        )

    def forward(self, x):
        # x shape: B, C, H, W -> B, dim, H / p, W / p -> B, (H * W / p / p), dim
        return self.patch_embed(x).transpose(1, 2)


class CondLayerNorm(nn.Module):
    def __init__(self, dim, num_classes, eps=1e-6, bias=True):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.bias = bias
        if self.bias: 
            self.embed = nn.Embedding(num_classes, 2 * dim)
        else:
            self.embed = nn.Embedding(num_classes, dim)
        self.embed.weight.data.zero_()

    def forward(self, x, noise_level):
        out = self.ln(x)
        if self.bias: 
            gamma, beta = self.embed(noise_level).unsqueeze(1).chunk(2, dim=-1)
            out = gamma * out + beta
        else:
            gamma = self.embed(noise_level).unsqueeze(1)
            out = gamma * out
        return out


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, qkv_bias=True, flash=True):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.attn = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_proj = nn.Linear(dim, dim)
        self.flash = flash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, dim = x.shape
        q, k, v = self.attn(x).chunk(3, -1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.flash:
            attn = F.scaled_dot_product_attention(q, k, v)
        else: 
            attn = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(q.size(-1)))
            attn = F.softmax(attn, dim=-1)
            attn = attn @ v
        attn = attn.transpose(1, 2).contiguous().view(B, T, dim)
        return self.attn_proj(attn)
    

class FeedForward(nn.Module):   
    def __init__(self, in_features: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NCSTBlock(nn.Module):
    def __init__(self, dim, n_heads, total_steps, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(dim, int(mlp_ratio * dim))
        self.adaLN_modulation = nn.Embedding(total_steps, 6 * dim)

    def forward(self, x, noise_level):
        scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = self.adaLN_modulation(noise_level).unsqueeze(1).chunk(6, dim=-1)
        x = x + gate_attn * self.attn(scale_attn * self.norm1(x) +  shift_attn)
        x = x + gate_mlp * self.mlp(scale_mlp * self.norm2(x) +  shift_mlp)
        return x
    

class NCST(nn.Module):
    def __init__(self, config: NCSTConfig):
        super().__init__()
        self.config = config
        self.seq_length = int(config.noise_dim[1] // config.patch_size) ** 2
        self.patch_proj = PatchEmbed(config.noise_dim[0], config.dim, config.patch_size) 
        self.pe = nn.Embedding(self.seq_length, config.dim)
        self.blocks = nn.ModuleList([NCSTBlock(config.dim, config.n_heads, config.total_steps) for _ in range(config.n_blocks)])
        self.norm = CondLayerNorm(config.dim, config.total_steps)
        hidden_dim = int(config.patch_size ** 2 * config.noise_dim[0])
        self.proj = nn.Linear(config.dim, hidden_dim)
        self.init_weights()

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        patch_embed = self.patch_proj(x)
        position_embed = self.pe(torch.arange(0, self.seq_length, device=x.device))
        z = patch_embed + position_embed
        for block in self.blocks:
            z = block(z, noise_level)
        z = self.norm(z, noise_level)
        # z shape B, T, dim
        z = self.unpatchify(self.proj(z))
        return z

    def unpatchify(self, x: torch.Tensor):
        # x shape: B, h*w, p*p*c (T = h*w)
        # img: B, c, h*p, w*p
        c = self.config.noise_dim[0]
        p = self.config.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.view((x.shape[0], h, w, p, p, c))
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(-1, c, h * p, w * p)
        return x
    
    def init_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_proj.patch_embed[0].weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_proj.patch_embed[0].bias.data, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation.weight, 0)