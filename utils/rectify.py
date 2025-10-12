import cv2
import numpy as np
from typing import Tuple, Optional, List

def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Accept pts shape (4,2). Return ordered points: [tl, tr, br, bl].
    Works even if input order is arbitrary.
    """
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _rect_size_from_quad(quad: np.ndarray) -> Tuple[int, int]:
    """Given 4 ordered pts (tl,tr,br,bl) compute target width & height (integers)."""
    tl, tr, br, bl = quad
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(round(max(heightA, heightB)))
    # ensure non-zero
    maxW = max(1, maxW)
    maxH = max(1, maxH)
    return maxW, maxH


def rectify_plate_from_obb(
    img: np.ndarray,
    obb_xyxyxyxy: np.ndarray,
    target_size: Optional[Tuple[int,int]] = None,
    pad: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectify (warp to top-down) a license plate using YOLO-OBB 4 corner coords.

    Args:
      img: source BGR or grayscale image (numpy array).
      obb_xyxyxyxy: tensor or array of shape (4, 2) containing corner coordinates
                    [x1,y1], [x2,y2], [x3,y3], [x4,y4]
      target_size: optional (width, height) to force output size. If None, computed
                   from quad geometry.
      pad: optional padding in pixels added around the computed rect before warping.

    Returns:
      (warped_plate, M) where warped_plate is the rectified image (numpy array)
      and M is the 3x3 homography matrix used.
    """
    # Convert to numpy if tensor
    if hasattr(obb_xyxyxyxy, 'cpu'):
        pts = obb_xyxyxyxy.cpu().numpy().reshape(4, 2)
    else:
        pts = np.asarray(obb_xyxyxyxy, dtype=float).reshape(4, 2)

    # If padding requested, expand points away from centroid
    if pad != 0:
        center = pts.mean(axis=0)
        vecs = pts - center
        # scale by (1 + pad / avg_dim)
        avg_dim = max(1.0, np.mean(np.linalg.norm(vecs, axis=1)))
        scale = 1.0 + (pad / avg_dim)
        pts = center + vecs * scale

    # Order points clockwise: tl, tr, br, bl
    try:
        quad = _order_points_clockwise(pts)
    except Exception:
        quad = pts.astype("float32")

    # Check degeneracy: area of quad
    def quad_area(q):
        x = q[:,0]
        y = q[:,1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area = quad_area(quad)
    if area < 1.0:
        # fallback: use minAreaRect
        rect = cv2.minAreaRect(pts.astype("float32"))
        box = cv2.boxPoints(rect)
        quad = _order_points_clockwise(box)

    # compute target size
    if target_size is None:
        maxW, maxH = _rect_size_from_quad(quad)
    else:
        maxW, maxH = int(target_size[0]), int(target_size[1])

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    # compute homography & warp
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC)

    return warped, M


def rectify_all_plates(img: np.ndarray, yolo_result, pad: int = 5) -> List[np.ndarray]:
    """
    Rectify all detected plates from YOLO OBB result.
    
    Args:
        img: source image (BGR)
        yolo_result: YOLO prediction result (e.g., plates[0])
        pad: padding around plate edges
        
    Returns:
        List of rectified plate images
    """
    rectified_plates = []
    
    if yolo_result.obb is None or len(yolo_result.obb) == 0:
        return rectified_plates
    
    # Get all OBB coordinates
    obb_coords = yolo_result.obb.xyxyxyxy  # shape: (N, 4, 2)
    
    for i, obb in enumerate(obb_coords):
        try:
            rectified, _ = rectify_plate_from_obb(img, obb, pad=pad)
            rectified_plates.append(rectified)
        except Exception as e:
            print(f"Failed to rectify plate {i}: {e}")
            continue
    
    return rectified_plates

