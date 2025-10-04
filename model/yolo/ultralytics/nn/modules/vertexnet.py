import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- small helpers ----
def conv_bn_act(c_in, c_out, k=3, s=1, p=None):
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True),
    )

class PNormSE(nn.Module):
    """
    Squeeze-Excite that concatenates GAP and GMP channels: [avg, max] -> 1x1 -> scale
    """
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(2 * c, mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Global Average & Max
        a = F.adaptive_avg_pool2d(x, 1)  # [B,C,1,1]
        m = F.adaptive_max_pool2d(x, 1)  # [B,C,1,1]
        s = torch.cat([a, m], dim=1)     # [B,2C,1,1]
        scale = self.fc(s)               # [B,C,1,1]
        return x * scale

class IntegrationBlock(nn.Module):
    """
    Residual: x -> 3x3 -> 3x3 -> SE -> +x
    Keep width constant (e.g., 128) for deeper stages (fast/tiny-target friendly).
    """
    def __init__(self, c, se=True):
        super().__init__()
        self.cv1 = conv_bn_act(c, c, 3, 1)
        self.cv2 = conv_bn_act(c, c, 3, 1)
        self.se = PNormSE(c) if se else nn.Identity()

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        y = self.se(y)
        return x + y

class Downsample(nn.Module):
    """
    Downsample by stride-2 conv (as in paper after the first pooling).
    """
    def __init__(self, c):
        super().__init__()
        self.cv = conv_bn_act(c, c, 3, 2)

    def forward(self, x):
        return self.cv(x)

class VertexBackbone(nn.Module):
    """
    Produce three outputs for YOLO head: P3, P4, P5 (strides 8/16/32).
    Layout (example):
      stem: 7x7 s=2 -> MaxPool s=2  (stride 4)
      stage3: Downsample -> IB x N3 (stride 8)  -> P3
      stage4: Downsample -> IB x N4 (stride 16) -> P4
      stage5: Downsample -> IB x N5 (stride 32) -> P5
    """
    def __init__(self, in_ch=3, width=128, se=True, n_blocks=(2, 3, 3)):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_act(in_ch, width, k=7, s=2, p=3),  # stride 2
            nn.MaxPool2d(kernel_size=2, stride=2)      # stride 4
        )

        def make_stage(n):
            return nn.Sequential(*[IntegrationBlock(width, se=se) for _ in range(n)])

        self.ds3 = Downsample(width)       # to stride 8
        self.stg3 = make_stage(n_blocks[0])

        self.ds4 = Downsample(width)       # to stride 16
        self.stg4 = make_stage(n_blocks[1])

        self.ds5 = Downsample(width)       # to stride 32
        self.stg5 = make_stage(n_blocks[2])

    def forward(self, x):
        x = self.stem(x)        # stride 4
        x = self.ds3(x)
        p3 = self.stg3(x)       # stride 8
        x = self.ds4(p3)
        p4 = self.stg4(x)       # stride 16
        x = self.ds5(p4)
        p5 = self.stg5(x)       # stride 32
        return [p3, p4, p5]

class Identity3(nn.Module):
    def forward(self, xs):  # xs is [P3,P4,P5]
        return xs