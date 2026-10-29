"""Pyramidal KLT with forward/backward checking and robust geometric fitting.
Tracks use a shared initial reference frame; replacement tracks are mapped into it.
Tracking loss is latched: explicitly reselect a clean page to recover.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from .geometry import project

@dataclass
class TrackResult:
    ok: bool
    H: object
    points: object
    count: int = 0
    inlier_ratio: float = 0.0
    error: float = 0.0
    reason: str = ""

class PageTracker:
    def __init__(self, config):
        self.cfg = config
        self.lost = True

    def initialize(self, frame, corners, H, canonical):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(corners), 255)
        mask = cv2.erode(mask, np.ones((7,7),np.uint8))
        p = cv2.goodFeaturesToTrack(gray, self.cfg.max_features, 0.01, 7, mask=mask)
        if p is None or len(p) < self.cfg.min_features:
            raise ValueError("Not enough page features. Add printed details or corner marks.")
        self.previous = gray
        self.points = p.reshape(-1,2)
        self.reference = self.points.copy()
        self.initial_H = H.copy()
        self.canonical = canonical
        self.reference_area = abs(cv2.contourArea(np.float32(corners)))
        self.initial_corners = np.float32(corners)
        self.lost = False

    def fail(self, reason):
        self.lost = True
        return TrackResult(False, None, np.empty((0,2)), reason=reason)

    def update(self, frame):
        if self.lost:
            return self.fail("Tracking lost: press R to select the clean page again")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p = self.points.astype(np.float32).reshape(-1,1,2)
        params = dict(winSize=(21,21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,0.01))
        q, status, _ = cv2.calcOpticalFlowPyrLK(self.previous,gray,p,None,**params)
        if q is None:
            return self.fail("Optical flow failed")
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray,self.previous,q,None,**params)
        if back is None:
            return self.fail("Reverse optical flow failed")
        good = (status.ravel()>0) & (reverse.ravel()>0)
        good &= np.linalg.norm(back.reshape(-1,2)-self.points,axis=1) < self.cfg.forward_backward_error
        good &= np.isfinite(q.reshape(-1,2)).all(axis=1)
        ref, now = self.reference[good], q.reshape(-1,2)[good]
        if len(now) < self.cfg.min_features:
            return self.fail("Too few visible page features")
        if self.cfg.model == "homography":
            T, inliers = cv2.findHomography(ref, now, cv2.RANSAC, self.cfg.ransac_error)
        else:
            A, inliers = cv2.estimateAffinePartial2D(ref, now, method=cv2.RANSAC,
                                                     ransacReprojThreshold=self.cfg.ransac_error)
            T = None if A is None else np.vstack([A, [0,0,1]])
        if T is None or inliers is None or not np.isfinite(T).all():
            return self.fail("Could not fit page transform")
        keep = inliers.ravel().astype(bool)
        count, ratio = int(keep.sum()), float(keep.mean())
        if count < self.cfg.min_features or ratio < self.cfg.min_inlier_ratio:
            return self.fail("Page motion has low consensus")
        coverage = cv2.contourArea(cv2.convexHull(ref[keep])) / self.reference_area
        if coverage < self.cfg.min_coverage:
            return self.fail("Visible features are too concentrated")
        corners = project(self.initial_corners,T)
        area_ratio = abs(cv2.contourArea(corners)) / self.reference_area
        edges = np.linalg.norm(corners-np.roll(corners,1,axis=0),axis=1)
        if (not cv2.isContourConvex(corners) or not 0.15 < area_ratio < 6
                or edges.min() < 10 or np.linalg.cond(T) > 1e9):
            return self.fail("Implausible page transform")
        H = T @ self.initial_H
        error = float(np.median(np.linalg.norm(project(ref[keep],T)-now[keep],axis=1)))
        self.reference, self.points = ref[keep], now[keep]
        self.previous = gray
        return TrackResult(True,H,self.points.copy(),count,ratio,error)

    def replenish(self, frame, H, blocked, observation, clean):
        """Replace lost tracks only on visible pixels agreeing with clean memory.

        New points are mapped into initial camera coordinates so they share the
        reference frame of surviving tracks. This improves longevity but can
        introduce some model drift; it is not recovery from complete loss.
        """
        if self.lost or len(self.points) >= self.cfg.max_features * 0.65:
            return
        a = cv2.GaussianBlur(observation,(5,5),0).astype(np.int16)
        b = cv2.GaussianBlur(clean,(5,5),0).astype(np.int16)
        agrees = np.max(np.abs(a-b),axis=2) < 25
        safe = ((~blocked) & agrees).astype(np.uint8)*255
        safe = cv2.erode(safe,np.ones((7,7),np.uint8))
        mask = cv2.warpPerspective(safe,H,(frame.shape[1],frame.shape[0]),flags=cv2.INTER_NEAREST)
        for point in self.points:
            cv2.circle(mask,tuple(np.rint(point).astype(int)),9,0,-1)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        new = cv2.goodFeaturesToTrack(gray,self.cfg.max_features-len(self.points),0.01,9,mask=mask)
        if new is None:
            return
        now = new.reshape(-1,2)
        reference = project(now,self.initial_H @ np.linalg.inv(H))
        self.points = np.vstack([self.points,now])
        self.reference = np.vstack([self.reference,reference])
