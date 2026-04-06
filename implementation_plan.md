# Fisheye License Plate Detection — Research Continuation Plan

## Background & Context

Your current experimental setup:
- **Model**: YOLOv11n-OBB for oriented license plate detection
- **Datasets**: `dataset_fisheye` (779 val images, 2320×2320px), `dataset_normal`, and `dataset_mixed`
- **Best fisheye run**: mAP@50 ≈ **0.860**, mAP@50-95 ≈ **0.689**, Precision ≈ **0.859**, Recall ≈ **0.818**
- **Best normal run**: mAP@50 ≈ **0.937**, mAP@50-95 ≈ **0.890**, Precision ≈ **0.931**, Recall ≈ **0.889**
- **Best mixed run**: mAP@50 ≈ **0.891**, mAP@50-95 ≈ **0.767**, Precision ≈ **0.900**, Recall ≈ **0.841**
- **Gap**: ~0.077 mAP@50 drop from normal → fisheye (significantly reduced from prior baseline)
- **Hyperparameters**: AdamW, lr0=0.005, lrf=0.05, cos_lr=True, batch=32, **imgsz=1024**, mosaic=0.5, close_mosaic=20, warmup_epochs=5, patience=30, epochs=100
- **Environment**: Ultralytics 8.4.30, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, CUDA available, `torchvision.ops.deform_conv2d` confirmed available
- **Labels**: YOLO OBB format (class x1 y1 x2 y2 x3 y3 x4 y4, normalized)

---

## Phase 1 — Zonal mAP Analysis (Immediate)

### Goal
Spatially decompose the fisheye mAP by dividing the image into radial zones (center vs. periphery) to understand **where** detection failures concentrate. This strengthens the paper's analysis of radial distortion effects.

### Approach
Since fisheye images are 2320×2320 (square), we define concentric radial zones from the image center:

| Zone | Radial Range (normalized) | Description |
|------|--------------------------|-------------|
| Center | 0.0 – 0.33 | Inner third — minimal distortion |
| Middle | 0.33 – 0.66 | Mid-ring — moderate distortion |
| Periphery | 0.66 – 1.0 | Outer ring — maximum barrel distortion |

For each ground-truth OBB, compute the centroid from its 4 corners and measure its normalized radial distance from the image center. Run inference and match predictions to GT per zone, then compute per-zone metrics.

### Proposed Changes

---

#### [NEW] [zonal_map_analysis.py](file:///home/vdlung/nas/vdlung/LocTH/omni2rect/zonal_map_analysis.py)

A self-contained script that:

1. **Loads the best fisheye model** from `runs/obb/<your_fisheye_run>/weights/best.pt`
2. **Iterates over fisheye val images** in `dataset_fisheye/images/val/`
3. **For each image**:
   - Reads the corresponding GT label file (OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4`)
   - Computes the centroid of each GT box: `cx = mean(x1,x2,x3,x4)`, `cy = mean(y1,y2,y3,y4)`
   - Calculates normalized radial distance: `r = sqrt((cx-0.5)² + (cy-0.5)²) / 0.5` (0=center, 1=corner)
   - Assigns each GT to a zone based on `r`
   - Runs model inference on the image
   - Matches predictions to GT using IoU (OBB-aware)
   - Accumulates TP/FP/FN per zone
4. **Computes per-zone AP** using the COCO-style interpolation
5. **Outputs a summary table** and **saves a visualization** (bar chart + polar heatmap)

**Key design decisions**:
- Use `model.predict()` with `conf=0.25, iou=0.7` (matching val defaults)
- OBB IoU matching via `shapely` Polygon intersection (or `probiou` if available)
- Radial distance computed from normalized coordinates (label format is already normalized)

---

#### [NEW] [zonal_map_visualization.py](file:///home/vdlung/nas/vdlung/LocTH/omni2rect/zonal_map_visualization.py)

Optional companion script for generating publication-quality figures:
- Bar chart: mAP@50 per zone (center / middle / periphery)
- Scatter plot: per-object detection confidence vs. radial distance
- Polar density heatmap of GT distribution across zones

---

## Phase 2 — DCNv2 Ablation Study (Next)

### Goal
Insert Deformable Convolutional Networks v2 (DCNv2) into the YOLO11n-OBB backbone at stages 3–4 (P4/P5 feature levels), retrain with identical hyperparameters, and measure mAP recovery on fisheye images. Even a 0.05–0.10 mAP improvement is a publishable contribution given the already-narrowed baseline gap.

### Architecture Analysis

The YOLO11n-OBB backbone uses a width multiplier of **0.25**, scaling base channels accordingly. The effective layer structure is:

```
Layer  0: Conv         3 → 16    (P1/2)     — Stage 1
Layer  1: Conv         16 → 32   (P2/4)     — Stage 1
Layer  2: C3k2         32 → 64              — Stage 2
Layer  3: Conv         64 → 64   (P3/8)     — Stage 2 downsample
Layer  4: C3k2         64 → 128             — Stage 3
Layer  5: Conv         128 → 128 (P4/16)    — Stage 3 downsample  ← DCNv2 HERE
Layer  6: C3k2         128 → 128            — Stage 4
Layer  7: Conv         128 → 256 (P5/32)    — Stage 4 downsample  ← DCNv2 HERE
Layer  8: C3k2         256 → 256            — Stage 4
Layer  9: SPPF         256 → 256
Layer 10: C2PSA        256 → 256
```

**Strategy**: Replace the downsampling `Conv` layers at **Layer 5** (P4/16, 128→128) and **Layer 7** (P5/32, 128→256) with `DCNv2` modules. These are the layers corresponding to "backbone stages 3–4" in the paper terminology. These layers are chosen because:
- Stages 3–4 process mid-to-high-level features where geometric distortion is most impactful
- Deformable convolutions can learn spatially-varying receptive fields to compensate for barrel distortion
- Early stages (1–2) process too-low-level features; replacing them has minimal effect

### Implementation Approach

> [!IMPORTANT]
> We must install ultralytics in **editable mode** to register the custom DCNv2 module in the framework's model parser. This means cloning the ultralytics repo and running `pip install -e .` instead of using the pip-installed version.

### Proposed Changes

---

### Component 1: DCNv2 Module

#### [NEW] [dcn_module.py](file:///home/vdlung/nas/vdlung/LocTH/omni2rect/dcn_module.py)

A self-contained DCNv2 module using `torchvision.ops.DeformConv2d`:

```python
class DCNv2(nn.Module):
    """Deformable Convolution v2 with learnable offsets and modulation masks.
    
    Compatible with Ultralytics YAML parser.
    Args: c1 (int), c2 (int), k (int)=3, s (int)=1, p (int)=None, g (int)=1
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        # Offset conv: predicts 2*k*k offsets per group
        self.offset_conv = nn.Conv2d(c1, 2 * k * k, k, s, p, bias=True)
        # Mask conv: predicts k*k modulation weights per group  
        self.mask_conv = nn.Conv2d(c1, k * k, k, s, p, bias=True)
        # Deformable conv
        self.dcn = torchvision.ops.DeformConv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        # Initialize offsets to zero (start as regular conv)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
    
    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        x = self.dcn(x, offset, mask)
        return self.act(self.bn(x))
```

---

### Component 2: Ultralytics Integration

> [!WARNING]  
> The approach below requires modifying the ultralytics source code. We will clone the repo and install in editable mode to avoid breaking your existing installation.

#### Step 1: Clone and install ultralytics in editable mode

```bash
cd /home/vdlung/nas/vdlung/LocTH/omni2rect
git clone https://github.com/ultralytics/ultralytics.git ultralytics_src
cd ultralytics_src
git checkout v8.4.30  # match your current version
pip install -e .
```

#### Step 2: Register DCNv2 in ultralytics source

#### [MODIFY] [conv.py](file:///home/vdlung/anaconda3/lib/python3.11/site-packages/ultralytics/nn/modules/conv.py)
- Add the `DCNv2` class definition at the end of the file
- Add `DCNv2` to `__all__`

#### [MODIFY] [__init__.py](file:///home/vdlung/anaconda3/lib/python3.11/site-packages/ultralytics/nn/modules/__init__.py)
- Import `DCNv2` from `.conv`

#### [MODIFY] [tasks.py](file:///home/vdlung/anaconda3/lib/python3.11/site-packages/ultralytics/nn/tasks.py)
- Add `DCNv2` to the `base_modules` frozenset in `parse_model()` (line ~1575)
- This ensures the YAML parser treats it the same as `Conv`: `c1, c2 = ch[f], args[0]`

---

### Component 3: Custom Model YAML

#### [NEW] [yolo11n-obb-dcn.yaml](file:///home/vdlung/nas/vdlung/LocTH/omni2rect/yolo11n-obb-dcn.yaml)

Modified YOLO11n-OBB config with DCNv2 at stages 3–4:

```yaml
# YOLO11n-OBB with DCNv2 at backbone stages 3-4
nc: 1
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4
  - [-1, 1, DCNv2, [512, 3, 2]]        # 5-P4/16  ← CHANGED from Conv
  - [-1, 2, C3k2, [512, True]]         # 6
  - [-1, 1, DCNv2, [1024, 3, 2]]       # 7-P5/32  ← CHANGED from Conv
  - [-1, 2, C3k2, [1024, True]]        # 8
  - [-1, 1, SPPF, [1024, 5]]           # 9
  - [-1, 2, C2PSA, [1024]]             # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]        # 13
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]        # 16 (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]        # 19 (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]        # 22 (P5/32-large)
  - [[16, 19, 22], 1, OBB, [nc, 1]]   # OBB(P3, P4, P5)
```

> [!NOTE]
> The YAML backbone channel values are **pre-scale** (before applying the `n` width multiplier of 0.25). The effective channels at runtime are 1/4 of the listed values (e.g. 512 → 128, 1024 → 256). This matches the architecture analysis table above.

---

### Component 4: Training Script

#### [NEW] [train_dcn_fisheye.py](file:///home/vdlung/nas/vdlung/LocTH/omni2rect/train_dcn_fisheye.py)

Training script with **identical hyperparameters** to the best fisheye run for a controlled ablation:

```python
"""Train YOLO11n-OBB-DCN on fisheye dataset.

DCN Ablation: Same hyperparameters as the best fisheye baseline run,
but with DCNv2 replacing Conv at backbone stages 3-4.
"""
from ultralytics import YOLO

model = YOLO("yolo11n-obb-dcn.yaml")  # build from custom YAML
# Load pretrained backbone weights from standard yolo11n-obb.pt
# (DCNv2 layers will be randomly initialized since they're new)

model.train(
    data="dataset_fisheye/data.yaml",
    epochs=100,
    imgsz=1024,
    device=0,
    batch=32,
    optimizer="AdamW",
    lr0=0.005,
    lrf=0.05,
    cos_lr=True,
    warmup_epochs=5,
    mosaic=0.5,
    close_mosaic=20,
    patience=30,
    project="runs/obb",
    name="train_dcn_fisheye",
)
```

---

### Component 5: Comparative Evaluation

#### [NEW] [compare_results.py](file:///home/vdlung/nas/vdlung/LocTH/omni2rect/compare_results.py)

Script to generate a publication-quality comparison table and plot:

| Model | Dataset | mAP@50 | mAP@50-95 | Precision | Recall | Params | GFLOPs |
|-------|---------|--------|-----------|-----------|--------|--------|--------|
| YOLOv11n-OBB (baseline) | Normal  | 0.937 | 0.890 | 0.931 | 0.889 | ~2.7M | ~6.5 |
| YOLOv11n-OBB (baseline) | Fisheye | 0.860 | 0.689 | 0.859 | 0.818 | ~2.7M | ~6.5 |
| YOLOv11n-OBB (baseline) | Mixed   | 0.891 | 0.767 | 0.900 | 0.841 | ~2.7M | ~6.5 |
| YOLOv11n-OBB + DCNv2    | Fisheye | TBD   | TBD   | TBD   | TBD   | ~2.9M | ~7.0 |

---

## Open Questions

> [!IMPORTANT]
> **Q1: Editable ultralytics installation** — Installing ultralytics in editable mode will replace your current pip-installed v8.4.30. Are you okay with this, or would you prefer we create a separate conda environment (e.g., `conda create -n dcn_env`) to isolate the changes?
> **Answer**: I have already set up ultralytics in editable mode in my virtual environment, so choose the .venv as the primary environment, but help me verify again.

> [!IMPORTANT]  
> **Q2: Zone definitions** — The plan uses 3 concentric zones (center/middle/periphery at 0.33 and 0.66 radial thresholds). Would you prefer:
> - (A) 3 zones as proposed
> - (B) 4 zones (quadrants: 0–0.25, 0.25–0.50, 0.50–0.75, 0.75–1.0)
> - (C) Custom thresholds based on your fisheye lens model?
> **Answer**: I choose option A (3 zones as proposed)

> [!WARNING]
> **Q3: Pretrained weights for DCN model** — When building from the custom YAML, only layers with matching architectures can load pretrained weights. The DCNv2 layers (replacing Conv at layers 5 and 7) will be randomly initialized. This is the expected behavior for an ablation study, but worth confirming you're okay with this approach vs. a more complex weight-transfer strategy.
> **Answer**: I agree with the originally proposed approach.

> [!NOTE]
> **Q4: Mixed dataset** — Results show the mixed-trained model achieves an intermediate mAP@50 of 0.891 on fisheye. Should Phase 2 also run a DCNv2 ablation on the mixed dataset for a more complete comparison, or is fisheye-only sufficient for the paper?
> **Answer**: I want to run this on fisheye-only images first before exploring other 2 options.

> [!NOTE]
> **Q5: Priority** — Should I start implementing Phase 1 (zonal analysis) immediately after approval, or do you want to review both phases and approve them together?
> **Answer**: I want Phase 1 to be implemented immediately after approval.

---

## Verification Plan

### Phase 1 Verification
- Run the zonal analysis script and verify it produces per-zone metrics
- Cross-check: sum of per-zone GT counts should equal total GT count in val set
- Validate that the overall mAP from zonal analysis ≈ 0.860 (matching the best fisheye run's reported mAP@50)

### Phase 2 Verification
- Verify custom YAML builds correctly: `model = YOLO("yolo11n-obb-dcn.yaml"); model.info()`
- Confirm DCNv2 layers appear in model summary at the correct positions (layers 5 and 7)
- Confirm parameter count increase is reasonable (~0.2M additional parameters from DCNv2 offset/mask convs)
- Run training and verify loss converges
- Compare final mAP against baseline fisheye run results

### Automated Tests
```bash
# Phase 1: Run zonal analysis
python zonal_map_analysis.py

# Phase 2: Verify model builds
python -c "from ultralytics import YOLO; m = YOLO('yolo11n-obb-dcn.yaml'); m.info()"

# Phase 2: Training
python train_dcn_fisheye.py

# Phase 2: Comparative evaluation
python compare_results.py
```
