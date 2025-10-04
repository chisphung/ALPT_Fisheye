# train_vertexnet.py
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

from backbone import VertexNet
from loss import VertexNetLoss

def _to_pixels(norm_xy, w, h):
    xy = norm_xy.clone()
    xy[:, 0] *= w
    xy[:, 1] *= h
    return xy

def _bbox_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def _xyxy_to_quad(xyxy):
    x1, y1, x2, y2 = xyxy.unbind(-1)
    # (x1,y1)-(x2,y1)-(x2,y2)-(x1,y2)
    return torch.stack([x1,y1, x2,y1, x2,y2, x1,y2], dim=-1)


class YOLOLPDataset(Dataset):
    """
    Expects an image directory with matching YOLO .txt files (same stem).
    Label lines per object:
      - BBox:     c cx cy w h
      - Quad8:    c x1 y1 x2 y2 x3 y3 x4 y4
    All coords normalized [0,1] in original image size.

    Returns per item:
      image:       FloatTensor [3, 256, 256]
      cls_target:  LongTensor  [N]
      box_target:  FloatTensor [N, 4] (xyxy in resized space)
      vert_target: FloatTensor [N, 8] (x1,y1,...,x4,y4 in resized space)
    """
    def __init__(self, img_dir, size=256, img_exts={".jpg",".jpeg",".png",".bmp"}):
        self.img_dir = Path(img_dir)
        self.size = size
        self.img_paths = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in img_exts])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        lbl_path = img_path.with_suffix(".txt")

        img = read_image(str(img_path)).float() / 255.0  # [C,H,W], uint8->float
        _, oh, ow = img.shape

        # Read labels (may be empty)
        cls_list, box_list, vert_list = [], [], []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    c = int(float(parts[0]))
                    nums = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)

                    if nums.numel() == 4:
                        # bbox cx cy w h (normalized)
                        bbox_norm = nums.view(1,4)  # [1,4]
                        # to pixels in original space
                        bbox_xyxy = _bbox_cxcywh_to_xyxy(bbox_norm).view(-1,2)
                        bbox_xyxy = _to_pixels(bbox_xyxy, ow, oh).view(-1)  # [4]
                        quad = _xyxy_to_quad(bbox_xyxy).view(-1)            # [8]
                    elif nums.numel() == 8:
                        # quad (x1 y1 ... x4 y4) normalized
                        quad_norm = nums.view(4,2)
                        quad_xy = _to_pixels(quad_norm, ow, oh)   # [4,2]
                        x = quad_xy[:,0]; y = quad_xy[:,1]
                        bbox_xyxy = torch.tensor([x.min(), y.min(), x.max(), y.max()], dtype=torch.float32)
                        quad = quad_xy.view(-1)  # [8]
                    else:
                        # Unsupported row; skip
                        continue

                    cls_list.append(c)
                    box_list.append(bbox_xyxy)
                    vert_list.append(quad)

        # Resize image to [size,size]
        img = TF.resize(img, [self.size, self.size])
        nh, nw = self.size, self.size

        # Scale boxes/verts from original->resized
        sx, sy = nw / ow, nh / oh
        if box_list:
            box_target = torch.stack(box_list, 0)  # [N,4]
            box_target[:, [0,2]] *= sx
            box_target[:, [1,3]] *= sy
            vert_target = torch.stack(vert_list, 0)  # [N,8]
            vert_target[:, 0::2] *= sx
            vert_target[:, 1::2] *= sy
            cls_target = torch.tensor(cls_list, dtype=torch.long)
        else:
            # No objects
            box_target = torch.zeros((0,4), dtype=torch.float32)
            vert_target = torch.zeros((0,8), dtype=torch.float32)
            cls_target = torch.zeros((0,), dtype=torch.long)

        return img, (cls_target, box_target, vert_target)

def detection_collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)  # [B,3,256,256]
    cls_list, box_list, vert_list = [], [], []
    for (cls_t, box_t, vert_t) in targets:
        cls_list.append(cls_t)    # [Ni]
        box_list.append(box_t)    # [Ni,4]
        vert_list.append(vert_t)  # [Ni,8]
    return images, (cls_list, box_list, vert_list)

# --- 2. Training Setup ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.08

model = VertexNet().to(DEVICE)
criterion = VertexNetLoss()
# The paper specifies using the SGD optimizer with momentum and weight decay 
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

# Cosine learning schedule with warm-up 
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=2000)

dataset = YOLOLPDataset(num_samples=100000)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Training Loop ---
print("Starting VertexNet training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        images = images.to(DEVICE)
        # Move targets to device and format them correctly
        # This step is highly dependent on your data loading and anchor matching logic
        
        optimizer.zero_grad()
        
        predictions = model(images)
        
        # The loss calculation requires matching predictions with targets
        # This is a complex part of object detection training, simplified here
        # loss = criterion(predictions, formatted_targets)
        loss = torch.tensor(0.1, requires_grad=True) # Placeholder loss
        
        loss.backward()
        optimizer.step()
        
        # Apply learning rate schedulers
        if i < 2000:
            warmup_scheduler.step()
        
        total_loss += loss.item()
    
    scheduler.step() # Cosine scheduler steps once per epoch

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"vertexnet_epoch_{epoch+1}.pth")

print("Training finished.")