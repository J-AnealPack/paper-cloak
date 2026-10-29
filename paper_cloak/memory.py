"""Conservative temporal clean-page memory; observation must be safe AND stable."""
import numpy as np
import cv2

class PageMemory:
    def __init__(self, initial, config):
        self.clean = initial.copy()
        self.previous = initial.copy()
        self.age = np.zeros(initial.shape[:2],np.uint16)
        self.cfg = config

    def update(self, observation, blocked, enabled):
        # Compare lightly smoothed images to tolerate subpixel warp jitter;
        # commit original pixels so the stored writing remains sharp.
        current = cv2.GaussianBlur(observation, (3, 3), 0)
        previous = cv2.GaussianBlur(self.previous, (3, 3), 0)
        delta = np.max(np.abs(current.astype(np.int16)-previous.astype(np.int16)),axis=2)
        stable = delta <= self.cfg.stable_difference
        safe = ~blocked & stable
        if not enabled:
            self.age.fill(0)
        else:
            self.age = np.where(safe,np.minimum(self.age.astype(np.int32)+1,65535),0).astype(np.uint16)
        commit = safe & (self.age >= self.cfg.confirmation_frames) & enabled
        self.clean[commit] = observation[commit]
        self.previous = observation.copy()
        return commit
