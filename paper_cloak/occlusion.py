"""Replaceable white-paper / dark-ink heuristic; NOT general object recognition.
A binary mask means 'do not learn these pixels'. Color thresholds are tunable.
"""
import cv2
import numpy as np

class PaperOcclusionDetector:
    def __init__(self, config):
        self.cfg = config

    def detect(self, page, valid):
        c = self.cfg
        hsv = cv2.cvtColor(page,cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(page,cv2.COLOR_BGR2GRAY)
        paper = (gray >= c.white_min) & (hsv[:,:,1] <= c.white_saturation_max)
        # Include intermediate gray antialiasing pixels: a threshold gap
        # otherwise labels letter edges as occluders and dilation hides the text.
        ink = (gray <= max(c.ink_max, c.white_min)) & (hsv[:,:,1] <= c.ink_saturation_max)
        # Reject whole dark components that contain a thick interior, while
        # permitting thin writing strokes. Fat marker strokes may be rejected.
        n, labels = cv2.connectedComponents(ink.astype(np.uint8))
        distance = cv2.distanceTransform(ink.astype(np.uint8),cv2.DIST_L2,5)
        radii = np.zeros(n,np.float32)
        np.maximum.at(radii,labels.ravel(),distance.ravel())
        thin_ink = ink & (radii[labels] <= c.max_ink_radius)
        blocked = ~(paper | thin_ink)
        if c.skin_enabled:
            ycc = cv2.cvtColor(page,cv2.COLOR_BGR2YCrCb)
            skin = ((ycc[:,:,1] >= c.skin_cr_min) & (ycc[:,:,1] <= c.skin_cr_max)
                    & (ycc[:,:,2] >= c.skin_cb_min) & (ycc[:,:,2] <= c.skin_cb_max))
            blocked |= skin
        r = c.obstruction_margin
        if r:
            blocked = cv2.dilate(blocked.astype(np.uint8),np.ones((2*r+1,2*r+1),np.uint8)) > 0
        blocked |= ~valid
        m = c.page_edge_margin
        if m:
            blocked[:m,:] = blocked[-m:,:] = True
            blocked[:,:m] = blocked[:,-m:] = True
        return blocked
