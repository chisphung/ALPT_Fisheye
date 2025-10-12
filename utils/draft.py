#!/usr/bin/env python3
import yaml
import os
import pathlib
import cv2
import numpy as np
import sys
import imghdr
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Validate image files referenced by a dataset YAML")
parser.add_argument("--dataset", "-d", default="/kaggle/working/dataset.yaml", help="path to dataset yaml")
parser.add_argument("--max-check", "-m", type=int, default=0, help="max number of files to check (0 = all)")
parser.add_argument("--show-sample", action="store_true", help="show a few valid image paths as sample")
args = parser.parse_args()

dataset_yaml = args.dataset

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def read_paths_from_text_file(p):
    ret = []
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for L in f:
            L = L.strip()
            if not L:
                continue
            # take first token in case of label lines like: /path/to/img.jpg x1 y1 x2 y2
            path_token = L.split()[0]
            ret.append(path_token)
    return ret

def collect_images(entry, yaml_parent):
    imgs = []
    if entry is None:
        return imgs
    # allow comma-separated strings
    if isinstance(entry, str) and "," in entry and not os.path.exists(entry):
        for item in entry.split(","):
            imgs += collect_images(item.strip(), yaml_parent)
        return imgs

    p = pathlib.Path(entry)
    # relative to yaml parent
    if not p.is_absolute():
        p = (pathlib.Path(yaml_parent) / p).resolve()

    if p.exists() and p.is_dir():
        for ext in IMAGE_EXTS:
            imgs += list(p.rglob(f"*{ext}"))
    elif p.exists() and p.is_file():
        suffix = p.suffix.lower()
        # If it's a common text/list file, parse it as paths
        if suffix in (".txt", ".lst", ".list"):
            items = read_paths_from_text_file(p)
            for it in items:
                # resolve relative to dataset yaml dir
                itp = pathlib.Path(it)
                if not itp.is_absolute():
                    itp = (pathlib.Path(dataset_yaml).parent / itp).resolve()
                imgs.append(itp)
        elif suffix in (".json",):
            # could be COCO; we won't parse full COCO here — add JSON path to list so user can check manually
            imgs.append(p)
        else:
            # treat as single image file
            imgs.append(p)
    else:
        # maybe a glob or missing path string (like '/some/path/*.jpg' or URL)
        s = str(entry)
        if "*" in s or "?" in s:
            # expand glob relative to yaml parent
            try:
                base = pathlib.Path(yaml_parent)
                for match in base.glob(s):
                    imgs.append(match.resolve())
            except Exception:
                pass
        else:
            # return as Path (will be checked for existence later)
            imgs.append(p)
    return imgs

def is_image_file_by_magic(path):
    try:
        t = imghdr.what(path)
        return t is not None
    except Exception:
        return False

def try_open_with_pil(path):
    try:
        with Image.open(path) as im:
            im.verify()  # verify checks for truncated/corrupt
        return True, None
    except Exception as e:
        return False, str(e)

def cv2_decode_ok(b):
    try:
        arr = np.frombuffer(b, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return (im is not None)
    except Exception:
        return False

def main():
    if not os.path.exists(dataset_yaml):
        print(f"Dataset yaml not found: {dataset_yaml}")
        sys.exit(2)

    with open(dataset_yaml, "r", encoding="utf-8", errors="ignore") as f:
        y = yaml.safe_load(f)

    all_imgs = []
    for key in ("train", "val", "test"):
        v = y.get(key)
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            for item in v:
                all_imgs += collect_images(item, pathlib.Path(dataset_yaml).parent)
        else:
            all_imgs += collect_images(v, pathlib.Path(dataset_yaml).parent)

    # dedupe and normalize paths
    seen = set()
    imgs = []
    for p in all_imgs:
        pp = pathlib.Path(p)
        if not pp.is_absolute():
            pp = (pathlib.Path(dataset_yaml).parent / pp).resolve()
        s = str(pp)
        if s not in seen:
            imgs.append(pp)
            seen.add(s)

    total = len(imgs)
    print(f"Found {total} image candidates to check.")
    if args.max_check and args.max_check > 0:
        imgs = imgs[: args.max_check]
        print(f"Limiting to first {len(imgs)} files for quick check (--max-check)")

    bad = []
    good_samples = []
    for i, p in enumerate(imgs, start=1):
        reason = None
        sp = str(p)
        if not p.exists():
            reason = "missing"
        else:
            try:
                sz = p.stat().st_size
            except Exception as e:
                reason = f"stat error: {e}"
                sz = None
            if reason is None:
                if sz == 0:
                    reason = "zero-byte"
                else:
                    # attempt read bytes and decode
                    try:
                        b = p.read_bytes()
                    except Exception as e:
                        reason = f"read error: {e}"
                    else:
                        # first try cv2.imdecode
                        if not cv2_decode_ok(b):
                            # check imghdr
                            if not is_image_file_by_magic(sp):
                                # try PIL to get more informative error
                                ok, pil_err = try_open_with_pil(sp)
                                if not ok:
                                    reason = f"not an image / corrupted (PIL: {pil_err})"
                                else:
                                    # PIL says ok but cv2 failed — still accept but warn
                                    good_samples.append(sp)
                            else:
                                # imghdr says it's image-ish, but cv2 failed — maybe unsupported format or partial
                                ok, pil_err = try_open_with_pil(sp)
                                if not ok:
                                    reason = f"corrupted (PIL: {pil_err})"
                                else:
                                    # PIL ok but cv2 decode fail -> warn (rare)
                                    # treat as OK but record as warning
                                    good_samples.append(sp)
                        else:
                            # cv2 decode ok
                            good_samples.append(sp)
        if reason:
            bad.append((sp, reason))
            print(f"[BAD] {sp} -> {reason}")
        if i % 200 == 0:
            sys.stdout.flush()

    if bad:
        print("\nSummary of bad files (first 100):")
        for p, reason in bad[:100]:
            print(p, "->", reason)
        print(f"... total bad: {len(bad)} / {total}")
        print("\nSuggested next steps:")
        print("- If many 'missing': check dataset.yaml paths (train/val) and whether text listing files have correct relative paths.")
        print("- If 'zero-byte' or 'corrupted': remove or re-export those images.")
        print("- If 'not an image / corrupted': open with an image viewer or re-export from source.")
        print("- To reproduce the error deterministically, try training with workers=0 (no multiprocessing).")
        sys.exit(1)
    else:
        print("All checked images decoded OK.")
        if args.show_sample:
            print("\nSample valid images:")
            for s in good_samples[:10]:
                print(" ", s)
        sys.exit(0)

if __name__ == "__main__":
    main()
