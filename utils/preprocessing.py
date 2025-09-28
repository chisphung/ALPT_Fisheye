import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Literal, Dict, Any

Point = Tuple[float, float]  # (x, y)
PlateType = Literal["one_line", "two_line"]

@dataclass
class VNPreprocessConfig:
    # Detector input (if you also resize for detection)
    det_short_side: int = 256

    # Recognition canvases
    out_size_one_line: Tuple[int, int] = (64, 256)   # (H, W)
    out_size_two_line: Tuple[int, int] = (96, 192)   # (H, W)

    # Decision threshold: classify two-line if H/W of the *quad* > this
    # (measured before rectification, using average edge lengths)
    two_line_hw_ratio_thresh: float = 0.45

    # Enlarge source quad to keep margins (like paper’s 1.25×)
    enlarge_ratio: float = 1.25

    # Warping params
    interp: int = cv2.INTER_LINEAR
    border_mode: int = cv2.BORDER_CONSTANT
    border_value: Tuple[int, int, int] = (0, 0, 0)

# ---------- geometry helpers ----------

def order_quad_pts_clockwise(pts: np.ndarray) -> np.ndarray:
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    order = np.argsort(ang)
    pts = pts[order]
    # rotate so we start at top-left (min x+y)
    s = np.sum(pts, axis=1)
    pts = np.roll(pts, -np.argmin(s), axis=0)
    return pts.astype(np.float32)

def enlarge_quad(quad: np.ndarray, ratio: float) -> np.ndarray:
    ctr = quad.mean(axis=0, keepdims=True)
    return ((quad - ctr) * ratio + ctr).astype(np.float32)

def avg_width_height_of_quad(quad: np.ndarray) -> Tuple[float, float]:
    # quad ordered TL, TR, BR, BL
    w_top  = np.linalg.norm(quad[1] - quad[0])
    w_bot  = np.linalg.norm(quad[2] - quad[3])
    h_left = np.linalg.norm(quad[3] - quad[0])
    h_right= np.linalg.norm(quad[2] - quad[1])
    return (0.5*(w_top + w_bot), 0.5*(h_left + h_right))

def classify_plate_type_by_hw(quad: np.ndarray, thresh: float) -> PlateType:
    w, h = avg_width_height_of_quad(quad)
    return "two_line" if (h / max(w, 1e-6)) > thresh else "one_line"

# ---------- rectification ----------

def rectify_plate(
    image_full: np.ndarray,
    verts_full: List[Point],
    cfg: VNPreprocessConfig
) -> Dict[str, Any]:
    """
    Returns:
      {
        'plate_type': 'one_line'|'two_line',
        'rectified': HxW np.uint8 image,
        'meta': { 'quad_ordered': np.ndarray (4,2), 'H_out': int, 'W_out': int }
      }
    """
    quad = order_quad_pts_clockwise(np.asarray(verts_full, dtype=np.float32))
    plate_type = classify_plate_type_by_hw(quad, cfg.two_line_hw_ratio_thresh)

    if plate_type == "one_line":
        H_out, W_out = cfg.out_size_one_line
    else:
        H_out, W_out = cfg.out_size_two_line

    # enlarge source quad before warping (safety margin)
    quad_src = enlarge_quad(quad, cfg.enlarge_ratio)

    dst = np.array([
        [0, 0],
        [W_out - 1, 0],
        [W_out - 1, H_out - 1],
        [0, H_out - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad_src, dst)
    rectified = cv2.warpPerspective(
        image_full, M, (W_out, H_out),
        flags=cfg.interp, borderMode=cfg.border_mode, borderValue=cfg.border_value
    )
    return {
        "plate_type": plate_type,
        "rectified": rectified,
        "meta": {"quad_ordered": quad, "H_out": H_out, "W_out": W_out}
    }

# ---------- optional: split 2-line plate into top/bottom lines ----------

def split_two_line(rectified_two_line: np.ndarray, split_bias: float = 0.52):
    """
    Split a rectified 2-line plate into top & bottom bands.
    `split_bias` lets you bias the horizontal split (top shorter in VN plates).
    Returns two images (top, bottom) resized to same width.
    """
    H, W = rectified_two_line.shape[:2]
    y = int(H * split_bias)  # e.g., 52% height for top band
    top  = rectified_two_line[:y, :]
    bot  = rectified_two_line[y:, :]

    # Normalize heights if your recognizer expects a fixed height per line
    H_line = 48
    top  = cv2.resize(top,  (W, H_line), interpolation=cv2.INTER_LINEAR)
    bot  = cv2.resize(bot,  (W, H_line), interpolation=cv2.INTER_LINEAR)
    return top, bot

# ---------- detector input (optional) ----------

def resize_with_aspect(image: np.ndarray, short_side: int):
    h, w = image.shape[:2]
    if h <= w:
        scale = short_side / float(h)
        new_h, new_w = short_side, int(round(w * scale))
    else:
        scale = short_side / float(w)
        new_w, new_h = short_side, int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale, 0, 0

def map_vertices_from_resized_to_original(verts_resized: List[Point], scale: float, pad_x=0, pad_y=0):
    v = np.asarray(verts_resized, dtype=np.float32)
    v[:, 0] -= pad_x; v[:, 1] -= pad_y
    return (v / scale).astype(np.float32)
