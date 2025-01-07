from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class UNetConfig:
    num_res_block: int = 2
    latent_ch: int = 4
    init_ch: int = 64
    ch_mults: tuple = (1, 2, 4, 8)
    total_steps: int = 10


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x, scale_shift=None):
        out = self.norm(x) # B, C, H, W
        if scale_shift is not None: 
            scale, shift = scale_shift
            out = out * (scale + 1) + shift
        out = self.silu(out)
        out = self.dropout(out)
        out = self.conv(out)
        return out


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.mlp = None
        if time_emb_dim is not None: 
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 2 * in_channels)
            )
        self.block0 = Block(in_channels, out_channels, dropout=0.1)
        self.block1 = Block(out_channels, out_channels)

        self.conv = None
        if in_channels != out_channels: 
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.in_channels = in_channels
    
    def forward(self, x, time_emb=None):
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expect input to have {self.in_channels} channels, got {C} instead"
        scale_shift = None
        if time_emb is not None and self.mlp is not None:
            scale_shift = self.mlp(time_emb)
            assert scale_shift.shape == (B, 2 * C), f"Expect scale_shift to have shape {B}, {2 * C}; got {scale_shift.shape} instead."
            scale_shift = scale_shift.view(B, 2 * C, 1, 1).chunk(2, dim=1)        
        out = self.block0(x, scale_shift)
        out = self.block1(out)
        if self.conv is not None:
            x = self.conv(x)
        return x + out


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.attn_proj = nn.Linear(channels, 3 * channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        x = self.norm(x)
        qkv = self.attn_proj(x.permute(0, 2, 3, 1).reshape(B, 1, H * W, C))
        q, k, v = qkv.chunk(3, dim=-1)
        # Flash attention
        attn = F.scaled_dot_product_attention(q, k, v).squeeze(1).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.out_proj(attn)


class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.size()
        out = self.conv(x)
        assert out.shape == (B, C, H//2, W//2)
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.size()
        out = F.interpolate(x, (H*2, W*2), mode='nearest')
        out = self.conv(out)
        assert out.shape == (B, C, H*2, W*2)
        return out
    

def get_time_embedding(time_step, embed_dim, base=10000, device=None):
    inv_freqs = 1.0 / base ** (torch.arange(0, embed_dim, 2, dtype=torch.float32, device=device).float() / embed_dim)
    emb = time_step[:, None] * inv_freqs[None, :] # T * embed // 2
    emb = torch.stack((torch.sin(emb), torch.cos(emb)), dim=-1).view(emb.shape[0], -1)
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class NCSN(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        self.num_res_block = config.num_res_block
        latent_ch = self.config.latent_ch
        init_ch = self.config.init_ch

        self.time_dim = 4 * init_ch
        self.time_embed = nn.Sequential(
            nn.Linear(init_ch, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        self.init_conv = nn.Conv2d(latent_ch, init_ch, kernel_size=3, padding=1)
        
        # All channels
        dims = [init_ch * m for m in self.config.ch_mults]
        # Down block
        self.downblock = nn.ModuleList()
        cur_dim = init_ch
        for idx, dim in enumerate(dims): 
            self.downblock.append(ResNetBlock(cur_dim, dim, self.time_dim))
            self.downblock.extend([ResNetBlock(dim, dim, self.time_dim) for _ in range(self.num_res_block - 1)])
            self.downblock.append(AttnBlock(dim))
            if idx != len(dims) - 1:
                self.downblock.append(DownSample(dim)) 
            cur_dim = dim

        self.midblock = nn.ModuleList([
            ResNetBlock(cur_dim, cur_dim, self.time_dim),
            AttnBlock(cur_dim),
            ResNetBlock(cur_dim, cur_dim, self.time_dim)
        ])
        
        self.upblock = nn.ModuleList()
        dims = dims[::-1]
        for idx, dim in enumerate(dims): 
            if idx != 0:
                self.upblock.append(nn.Conv2d(cur_dim, dim, kernel_size=3, padding=1))
            if idx == 0:
                self.upblock.append(ResNetBlock(cur_dim * 2, dim, self.time_dim))
            else: 
                self.upblock.append(ResNetBlock(cur_dim, dim, self.time_dim))
            self.upblock.extend([ResNetBlock(dim, dim, self.time_dim) for _ in range(self.num_res_block - 1)])
            self.upblock.append(AttnBlock(dim))
            if idx != len(dims) - 1:
                self.upblock.append(UpSample(dim)) 
            cur_dim = dim
        
        self.last_norm = nn.GroupNorm(num_channels=init_ch, num_groups=32, eps=1e-6)
        self.last_conv = nn.Conv2d(init_ch, latent_ch, kernel_size=3, padding=1)
        
    def forward(self, x, time):
        tembed = get_time_embedding(time, self.config.init_ch, device=x.device)
        time_embed = self.time_embed(tembed)
        z = self.init_conv(x)
        downs = []
        for layer in self.downblock:
            if isinstance(layer, ResNetBlock):
                z = layer(z, time_embed)
            else:
                z = layer(z)
                if isinstance(layer, AttnBlock):
                    downs.append(z)
        for layer in self.midblock:
            if (isinstance(layer, ResNetBlock)):
                z = layer(z, time_embed)
            else:
                z = layer(z)
        idx = 0 
        for layer in self.upblock:
            if (isinstance(layer, ResNetBlock)):
                if idx % self.num_res_block == 0:
                    z = torch.cat((z, downs.pop()), dim=1)
                z = layer(z, time_embed)
                idx += 1
            else:
                z = layer(z)
        z = self.last_conv(self.last_norm(z))
        return z