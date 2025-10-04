import torch
import torch.nn as nn
import torch.nn.functional as F

# p-norm SE Module [cite: 241]
class PNormSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super(PNormSE, self).__init__()
        # Excitation step [cite: 259]
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, c, _, _ = x.size()
        
        # Squeeze step: 1-norm (avg pool) and infinity-norm (max pool) [cite: 253, 254]
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        
        # Concatenate features [cite: 258, 259]
        y = torch.cat([avg_pool, max_pool], dim=1)
        
        y = self.fc(y)
        return x * y

# Integration Block (IB) [cite: 239]
class IntegrationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(IntegrationBlock, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.p_norm_se = PNormSE(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.residual_branch(x)
        res = self.p_norm_se(res)
        return F.relu(res + self.shortcut(x))

# VertexNet Model
class VertexNet(nn.Module):
    def __init__(self, num_anchors=9, num_classes=2):
        super(VertexNet, self).__init__()
        
        # Backbone Network [cite: 224]
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = self._make_stage(128, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 128, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(128, 128, num_blocks=2, stride=2)
        self.stage5 = self._make_stage(128, 128, num_blocks=2, stride=2)
        self.stage6 = self._make_stage(128, 128, num_blocks=2, stride=2)

        # Fusion Network (FPN-like) [cite: 271]
        self.p5_fusion = self._make_fusion_layer(128, 128)
        self.p4_fusion = self._make_fusion_layer(128, 128)
        self.p3_fusion = self._make_fusion_layer(128, 128)

        # Head Network [cite: 278, 279]
        output_channels = num_anchors * (num_classes + 4 + 8) # 2 scores, 4 box offsets, 8 vertex offsets
        self.shared_head_A = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        self.shared_head_B = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = [IntegrationBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(IntegrationBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_fusion_layer(self, ch1, ch2):
        return nn.Sequential(
            nn.Conv2d(ch1 + ch2, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Backbone
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2) # Output for fusion P3
        c4 = self.stage4(c3) # Output for fusion P4
        c5 = self.stage5(c4) # Output for fusion P5
        c6 = self.stage6(c5) # Output for head

        # Fusion
        p5 = self.p5_fusion(torch.cat([c5, F.interpolate(c6, scale_factor=2, mode='bilinear')], dim=1))
        p4 = self.p4_fusion(torch.cat([c4, F.interpolate(p5, scale_factor=2, mode='bilinear')], dim=1))
        p3 = self.p3_fusion(torch.cat([c3, F.interpolate(p4, scale_factor=2, mode='bilinear')], dim=1))

        # Head
        pred_p3 = self.shared_head_A(p3)
        pred_p4 = self.shared_head_A(p4)
        pred_p5 = self.shared_head_B(p5)
        pred_c6 = self.shared_head_B(c6)

        return [pred_p3, pred_p4, pred_p5, pred_c6]