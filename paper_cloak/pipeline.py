"""Coordinates independent tracking, segmentation, memory and rendering modules."""
from dataclasses import dataclass
import numpy as np
from .geometry import page_geometry, rectify, composite
from .tracking import PageTracker
from .occlusion import PaperOcclusionDetector
from .memory import PageMemory
from .config import Config

@dataclass
class FrameResult:
    output: object
    normalized: object
    clean: object
    blocked: object
    committed: object
    tracking: object

class PaperCloak:
    def __init__(self, config=None, tracker=None, detector=None):
        self.cfg = config or Config()
        self.tracker = tracker or PageTracker(self.cfg)
        self.detector = detector or PaperOcclusionDetector(self.cfg)
        self.ready = False

    def initialize(self, frame, corners):
        self.size,self.canonical,H = page_geometry(corners,self.cfg.page_width)
        initial,valid = rectify(frame,H,self.size)
        if valid.mean() < 0.995:
            raise ValueError("Select a page entirely inside the image")
        self.tracker.initialize(frame,corners,H,self.canonical)
        self.memory = PageMemory(initial,self.cfg)
        self.ready = True

    def process(self, frame):
        if not self.ready:
            raise RuntimeError("Call initialize with an unobstructed page first")
        track = self.tracker.update(frame)
        if not track.ok:
            self.memory.age.fill(0)
            # Never draw stale geometry over a page whose location is unknown.
            return FrameResult(frame.copy(),None,self.memory.clean.copy(),None,None,track)
        page,valid = rectify(frame,track.H,self.size)
        blocked = self.detector.detect(page,valid)
        if hasattr(self.tracker, "replenish"):
            self.tracker.replenish(frame,track.H,blocked,page,self.memory.clean)
        committed = self.memory.update(page,blocked,self.cfg.dynamic)
        out = composite(frame,self.memory.clean,track.H)
        return FrameResult(out,page,self.memory.clean.copy(),blocked,committed,track)
