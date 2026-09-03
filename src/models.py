"""Data models and schemas for Kaplan-Meier curve digitization and IPD reconstruction."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class CalibrationConfig:
    """Configuration for pixel-to-data coordinate transformation."""
    # Pixel coordinates
    x_orig: float  # Pixel X corresponding to t_min
    y_orig: float  # Pixel Y corresponding to S_min (usually bottom of survival axis)
    x_max: float   # Pixel X corresponding to t_max
    y_max: float   # Pixel Y corresponding to S_max (usually top of survival axis)
    
    # Real data values
    t_min: float = 0.0
    t_max: float = 60.0
    s_min: float = 0.0
    s_max: float = 1.0  # Normalized to 0.0 - 1.0
    time_unit: str = "months"
    survival_scale: str = "0-1"  # "0-1" or "0-100"

    def validate(self) -> List[str]:
        errors = []
        if self.x_max <= self.x_orig:
            errors.append(f"X maximum pixel ({self.x_max}) must be greater than X origin pixel ({self.x_orig}).")
        if self.y_orig <= self.y_max:
            # Note: in images, y increases downward, so y_orig (S=0) should have larger pixel Y than y_max (S=1)
            errors.append(f"Y origin pixel ({self.y_orig}) should typically be greater than Y max pixel ({self.y_max}) because image coordinates start at top-left.")
        if self.t_max <= self.t_min:
            errors.append(f"Maximum time ({self.t_max}) must be greater than minimum time ({self.t_min}).")
        if self.s_max <= self.s_min:
            errors.append(f"Maximum survival ({self.s_max}) must be greater than minimum survival ({self.s_min}).")
        return errors


@dataclass
class DigitizedCurvePoint:
    """Individual digitized curve coordinate point."""
    arm_id: int
    arm_name: str
    time: float
    survival: float
    survival_scale: str
    pixel_x: float
    pixel_y: float
    source_image: str
    extraction_method: str = "auto_color_segmentation"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskInterval:
    """Number-at-risk interval data."""
    interval_idx: int
    t_risk: float
    n_risk: int
    lower_idx: int = 0
    upper_idx: int = 0


@dataclass
class PseudoIPDRow:
    """Reconstructed individual patient data row."""
    patient_id: str
    arm_id: int
    arm_name: str
    time: float
    event: int  # 1 = event, 0 = censored
    reconstruction_method: str = "Guyot et al. 2012"
    total_events_constraint: str = "none"  # "none" or str(reported_count)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QCReport:
    """Quality control evaluation results comparing reconstructed vs digitized data."""
    extracted_points_per_arm: Dict[str, int] = field(default_factory=dict)
    number_at_risk_used: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    estimated_events_per_arm: Dict[str, int] = field(default_factory=dict)
    estimated_censored_per_arm: Dict[str, int] = field(default_factory=dict)
    reconstructed_sample_size_per_arm: Dict[str, int] = field(default_factory=dict)
    max_absolute_km_error: Dict[str, float] = field(default_factory=dict)
    mean_absolute_km_error: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
