import os, glob
from PIL import Image

IMG_DIR = "/content/drive/MyDrive/UFPR-ALPR dataset/testing/clustered_images/"  # where images live
LBL_DIR = "/content/drive/MyDrive/UFPR-ALPR dataset/testing/extracted_labels/testing/"   # where labels live
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# If you ALSO want to convert detect-format YOLO (class cx cy w h) into axis-aligned OBB,
# set this True. Otherwise we leave detect labels unchanged.
MAKE_OBB_FROM_DETECT = False

def find_image(lbl_path):
    stem = os.path.splitext(os.path.basename(lbl_path))[0]
    # labels may be in a separate folder; we scan IMG_DIR for a matching basename
    for root, _, files in os.walk(IMG_DIR):
        for f in files:
            n, e = os.path.splitext(f)
            if n == stem and e.lower() in IMG_EXTS:
                return os.path.join(root, f)
    return None

def clamp01(v): 
    return 0.0 if v < 0 else 1.0 if v > 1 else v

converted = skipped = 0
for lbl_path in glob.glob(os.path.join(LBL_DIR, "**/*.txt"), recursive=True):
    img_path = find_image(lbl_path)
    if not img_path:
        print(f"[skip] no image for {lbl_path}")
        skipped += 1
        continue

    with Image.open(img_path) as im:
        W, H = im.size

    out = []
    changed = False
    with open(lbl_path, "r", encoding="utf-8") as f:
        for raw in (ln.strip() for ln in f if ln.strip()):
            parts = raw.split()
            if len(parts) < 2: 
                continue
            cls = parts[0]
            nums = parts[1:]

            # ---- Case A: OBB (class + 8 coords)
            if len(nums) == 8:
                try:
                    vals = [float(x) for x in nums]
                except:
                    print(f"[warn] non-numeric OBB in {lbl_path}: {raw}")
                    continue

                # Decide if pixel or already normalized:
                # If any coord > 1.5, we treat as pixel coordinates.
                if any(abs(v) > 1.5 for v in vals):
                    # normalize x by W and y by H, preserving order x1 y1 x2 y2 ...
                    norm = []
                    for i, v in enumerate(vals):
                        if i % 2 == 0:   # x
                            norm.append(clamp01(v / W))
                        else:            # y
                            norm.append(clamp01(v / H))
                    out.append(" ".join([cls] + [f"{v:.6f}" for v in norm]))
                    changed = True
                else:
                    # Already normalized [0,1] — keep
                    out.append(raw)

            # ---- Case B: Detect (class + cx cy w h). By default we leave as-is.
            elif len(nums) == 4:
                if not MAKE_OBB_FROM_DETECT:
                    out.append(raw)  # unchanged
                else:
                    # Optional: convert detect → axis-aligned OBB
                    try:
                        cx, cy, w, h = map(float, nums)
                    except:
                        print(f"[warn] non-numeric detect in {lbl_path}: {raw}")
                        continue
                    # If looks like pixels, normalize first
                    if max(abs(cx), abs(cy), abs(w), abs(h)) > 1.5:
                        cx, cy, w, h = cx / W, cy / H, w / W, h / H
                    w = max(1e-6, min(1.0, w)); h = max(1e-6, min(1.0, h))
                    cx = clamp01(cx); cy = clamp01(cy)
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy - h/2
                    x3, y3 = cx + w/2, cy + h/2
                    x4, y4 = cx - w/2, cy + h/2
                    out.append(f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}")
                    changed = True

            else:
                print(f"[warn] Unexpected field count ({len(nums)}) in {lbl_path}: {raw}")

    if not out:
        skipped += 1
        continue

    # Backup once
    bak = lbl_path + ".bak"
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as b:
            with open(lbl_path, "r", encoding="utf-8") as src:
                b.write(src.read())

    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    converted += 1 if changed else 0

print(f"Done. Files_changed={converted}, Files_unchanged={len(list(glob.iglob(os.path.join(LBL_DIR, '**/*.txt'), recursive=True))) - converted}, Skipped={skipped}")
