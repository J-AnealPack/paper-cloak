"""Coordinate convention: H maps canonical page pixels to camera pixels."""
import cv2
import numpy as np

def project(points, H):
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)

def page_geometry(corners, width):
    p = np.asarray(corners, np.float32)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError("Supply four finite corners: TL, TR, BR, BL")
    contour = p.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 400:
        raise ValueError("Corners must describe a convex page in perimeter order")
    w = (np.linalg.norm(p[1]-p[0]) + np.linalg.norm(p[2]-p[3])) / 2
    h = (np.linalg.norm(p[3]-p[0]) + np.linalg.norm(p[2]-p[1])) / 2
    height = int(round(width*h/w))
    if not 100 <= height <= 2400:
        raise ValueError("Page aspect ratio is too extreme")
    canonical = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
    return (width, height), canonical, cv2.getPerspectiveTransform(canonical, p)

def rectify(frame, H, size):
    inv = np.linalg.inv(H)
    page = cv2.warpPerspective(frame, inv, size)
    valid = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), inv, size,
                                flags=cv2.INTER_NEAREST) > 0
    return page, valid

def composite(frame, page, H):
    size = (frame.shape[1], frame.shape[0])
    warped = cv2.warpPerspective(page, H, size)
    mask = cv2.warpPerspective(np.full(page.shape[:2], 255, np.uint8), H, size,
                              flags=cv2.INTER_NEAREST)
    # Erode one pixel to avoid interpolated black borders.
    mask = cv2.erode(mask, np.ones((3,3), np.uint8)) > 0
    out = frame.copy()
    out[mask] = warped[mask]
    return out
