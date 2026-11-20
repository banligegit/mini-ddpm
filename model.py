"""Small U-Net with optional class conditioning. Run: python model.py."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('freq', torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2 - 1)))

    def forward(self, t):
        a = t.float()[:, None] * self.freq[None]
        return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, td):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(td, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(temb))[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class TinyUNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28, skip connections preserve digit details."""
    def __init__(self, base=32, time_dim=128, num_classes=None):
        super().__init__()
        if base % 8:
            raise ValueError('base must be divisible by 8 for GroupNorm')
        if num_classes is not None and num_classes < 1:
            raise ValueError('num_classes must be positive')
        self.num_classes = num_classes
        self.time = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, time_dim)
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.enc1 = ResBlock(base, base, time_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc2 = ResBlock(base * 2, base * 2, time_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.mid1 = ResBlock(base * 4, base * 4, time_dim)
        self.mid2 = ResBlock(base * 4, base * 4, time_dim)
        self.up2 = nn.Conv2d(base * 4, base * 2, 3, padding=1)
        self.dec2 = ResBlock(base * 4, base * 2, time_dim)
        self.up1 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.dec1 = ResBlock(base * 2, base, time_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x, t, y=None):
        temb = self.time(t)
        if self.num_classes is not None:
            if y is None or y.shape != (x.shape[0],) or y.dtype != torch.long:
                raise ValueError('Conditional labels must be a long tensor of shape [batch]')
            if y.device != x.device or bool(((y < 0) | (y >= self.num_classes)).any()):
                raise ValueError('Labels must be on the image device and within the class range')
            # Both embeddings are [batch,128]; every residual block receives them.
            temb = temb + self.class_embedding(y)
        elif y is not None:
            raise ValueError('Unconditional models do not accept class labels')
        h1 = self.enc1(self.stem(x), temb)
        h2 = self.enc2(self.down1(h1), temb)
        h = self.mid2(self.mid1(self.down2(h2), temb), temb)
        h = self.up2(F.interpolate(h, size=h2.shape[-2:], mode='nearest'))
        h = self.dec2(torch.cat([h, h2], dim=1), temb)
        h = self.up1(F.interpolate(h, size=h1.shape[-2:], mode='nearest'))
        return self.out(self.dec1(torch.cat([h, h1], dim=1), temb))


if __name__ == '__main__':
    net = TinyUNet()
    print('Parameters:', sum(p.numel() for p in net.parameters()))
    print('Output:', net(torch.randn(2, 1, 28, 28), torch.tensor([0, 999])).shape)
