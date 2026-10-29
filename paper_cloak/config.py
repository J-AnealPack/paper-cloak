"""All tunable parameters; JSON keys correspond to these field names."""
from dataclasses import dataclass, asdict
import json

@dataclass
class Config:
    model: str = "similarity"  # or homography
    page_width: int = 600
    max_features: int = 350
    min_features: int = 10
    forward_backward_error: float = 1.5
    ransac_error: float = 3.0
    min_inlier_ratio: float = 0.55
    min_coverage: float = 0.08
    dynamic: bool = False
    confirmation_frames: int = 8
    stable_difference: int = 12
    white_min: int = 145
    white_saturation_max: int = 65
    ink_max: int = 115
    ink_saturation_max: int = 100
    max_ink_radius: float = 3.5
    skin_enabled: bool = True
    skin_cr_min: int = 135
    skin_cr_max: int = 180
    skin_cb_min: int = 85
    skin_cb_max: int = 135
    obstruction_margin: int = 7
    page_edge_margin: int = 5

    def __post_init__(self):
        if self.model not in ("similarity", "homography"):
            raise ValueError("model must be similarity or homography")
        if self.page_width < 100 or self.confirmation_frames < 1:
            raise ValueError("page_width >= 100 and confirmation_frames >= 1 required")
        if self.min_features < 4 or self.max_features < self.min_features:
            raise ValueError("Need at least four features and max_features >= min_features")
        if self.obstruction_margin < 0 or self.page_edge_margin < 0:
            raise ValueError("Margins cannot be negative")

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
