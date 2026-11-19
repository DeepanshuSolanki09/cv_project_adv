"""Unimodal branch models.  Class order everywhere: Negative, Neutral, Positive."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGFNN(nn.Module):
    """Feed-forward network: in -> 256 -> 128 -> 3 (ReLU, dropout 0.3)."""
    def __init__(self, n_in=750, n_cls=3, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, n_cls))

    def forward(self, x):
        return self.net(x)  # logits; softmax applied outside


class DropBlock2D(nn.Module):
    """DropBlock (Ghiasi et al., 2018): drops contiguous b x b regions during training."""
    def __init__(self, block=3, keep_prob=0.9):
        super().__init__()
        self.block, self.keep = block, keep_prob

    def forward(self, x):
        if not self.training or self.keep >= 1.0:
            return x
        _, _, h, w = x.shape
        gamma = (1 - self.keep) / self.block ** 2 * (h * w) / ((h - self.block + 1) * (w - self.block + 1))
        mask = (torch.rand_like(x[:, :1]) < gamma).float()
        mask = F.max_pool2d(mask, self.block, stride=1, padding=self.block // 2)
        mask = 1 - mask
        return x * mask * mask.numel() / mask.sum().clamp(min=1.0)


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FaceVGG(nn.Module):
    """Modified VGG: 3 stages of [conv-BN-ReLU] x2 + max-pool, DropBlock, dense head.
    width=64 reproduces the original report's 64/128/256 design; width=32 halves every
    stage (used in the executed experiment because only one CPU core was available)."""
    def __init__(self, n_cls=3, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_bn(1, w), conv_bn(w, w), nn.MaxPool2d(2),
            conv_bn(w, 2 * w), conv_bn(2 * w, 2 * w), nn.MaxPool2d(2),
            conv_bn(2 * w, 4 * w), conv_bn(4 * w, 4 * w), DropBlock2D(3, 0.9), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * w * 6 * 6, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.5), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


class SpeechCNN(nn.Module):
    """2D CNN on a 40 x 128 MFCC map: 3 conv blocks (32/64/128) + GAP + dense head."""
    def __init__(self, n_cls=3):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(1, 32), nn.MaxPool2d(2), conv_bn(32, 64), nn.MaxPool2d(2),
            conv_bn(64, 128), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.4), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18_feature_extractor(weights_path=None):
    """ImageNet-pretrained ResNet-18 with the 1000-way FC removed (512-d output).
    Weights: timm 'resnet18_a1_0-d63eafa0.pth' (same key layout as torchvision)."""
    import torchvision
    m = torchvision.models.resnet18(weights=None)
    if weights_path:
        sd = torch.load(weights_path, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not k.startswith("fc.")], missing
    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
