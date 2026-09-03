"""Axis calibration and coordinate transformation between pixel coordinates and KM survival values."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from src.models import CalibrationConfig


class AxisCalibrator:
    """Handles calibration, pixel-to-data mapping, and automated detection of x and y axes."""

    def __init__(self, config: CalibrationConfig):
        self.config = config

    def pixel_to_data(self, px: float, py: float, clamp: bool = True) -> Tuple[float, float]:
        """
        Convert pixel coordinates (px, py) to real-world (time, survival) coordinates.
        y-axis is strictly 0.0 to 1.0:
          py = y_orig corresponds to S = 0.0
          py = y_max corresponds to S = 1.0
        """
        cfg = self.config
        x_span = cfg.x_max - cfg.x_orig
        y_span = cfg.y_orig - cfg.y_max

        if abs(x_span) < 1e-6:
            t = cfg.t_min
        else:
            t = cfg.t_min + ((px - cfg.x_orig) / x_span) * (cfg.t_max - cfg.t_min)

        if abs(y_span) < 1e-6:
            s = 0.0
        else:
            s = (cfg.y_orig - py) / y_span

        if clamp:
            t = max(cfg.t_min, min(cfg.t_max, t))
            s = max(0.0, min(1.0, s))

        return float(t), float(s)

    def data_to_pixel(self, t: float, s: float) -> Tuple[float, float]:
        """Convert (time, survival) to pixel (px, py), with survival strictly in 0.0 - 1.0."""
        cfg = self.config
        t_span = cfg.t_max - cfg.t_min
        y_span = cfg.y_orig - cfg.y_max

        if abs(t_span) < 1e-6:
            px = cfg.x_orig
        else:
            px = cfg.x_orig + ((t - cfg.t_min) / t_span) * (cfg.x_max - cfg.x_orig)

        # s is in 0.0 to 1.0
        s_clamped = max(0.0, min(1.0, s))
        py = cfg.y_orig - s_clamped * y_span

        return float(px), float(py)

    @staticmethod
    def auto_detect_plot_bounds(image_bgr: np.ndarray) -> Dict[str, float]:
        """
        Robust automated detection of Kaplan-Meier plot axes:
        - Detects vertical Y-axis line on the left side of the plot.
        - Detects horizontal X-axis line at the base (S = 0.0) above risk table.
        - Detects X-max extent along the horizontal axis.
        - Detects Y-max extent (S = 1.0) along the vertical axis.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bin_img = (gray < 165).astype(np.uint8) * 255

        # -------------------------------------------------------------
        # 1. Vertical Y-axis detection
        # -------------------------------------------------------------
        v_len = max(25, int(h * 0.20))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
        v_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, v_kernel)
        v_sums = np.sum(v_lines > 0, axis=0)

        cand_xs = [x for x in range(int(w * 0.04), int(w * 0.42)) if v_sums[x] > v_len * 0.45]
        if cand_xs:
            best_x = cand_xs[0]
            max_v_span = 0
            for x in cand_xs:
                col = (gray[:, max(0, x - 1):min(w, x + 2)] < 165)
                col_has_line = np.any(col, axis=1)
                runs = np.diff(np.where(np.concatenate(([col_has_line[0]], col_has_line[:-1] != col_has_line[1:], [True])))[0])[::2]
                span = max(runs) if len(runs) > 0 else 0
                if span > max_v_span:
                    max_v_span = span
                    best_x = x
            x_orig = float(best_x)
        else:
            x_orig = float(int(w * 0.12))

        # -------------------------------------------------------------
        # 2. Horizontal X-axis detection (at S = 0.0)
        # -------------------------------------------------------------
        h_len = max(30, int(w * 0.20))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        h_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, h_kernel)
        h_sums = np.sum(h_lines > 0, axis=1)

        cand_ys = [y for y in range(int(h * 0.42), int(h * 0.90)) if h_sums[y] > h_len * 0.45]
        if cand_ys:
            best_y = cand_ys[0]
            max_h_span = 0
            best_x_max = x_orig + w * 0.75
            for y in cand_ys:
                row = np.any(gray[max(0, y - 1):min(h, y + 2), int(x_orig):int(w * 0.98)] < 165, axis=0)
                gaps = 0
                length = 0
                for idx, val in enumerate(row):
                    if val:
                        length = idx
                        gaps = 0
                    else:
                        gaps += 1
                        if gaps > 12:
                            break
                if length > max_h_span:
                    max_h_span = length
                    best_y = y
                    best_x_max = x_orig + length
            y_orig = float(best_y)
            x_max = float(best_x_max)
        else:
            y_orig = float(int(h * 0.75))
            x_max = float(min(w - 15, int(x_orig + w * 0.75)))

        # -------------------------------------------------------------
        # 3. Detect Y-max (top of survival axis at S = 1.0)
        # -------------------------------------------------------------
        col_up = np.any(gray[10:int(y_orig) + 1, max(0, int(x_orig) - 2):min(w, int(x_orig) + 3)] < 165, axis=1)
        gaps = 0
        top_idx = len(col_up) - 1
        for idx in range(len(col_up) - 1, -1, -1):
            if col_up[idx]:
                top_idx = idx
                gaps = 0
            else:
                gaps += 1
                if gaps > 14:
                    break
        y_max = float(10 + top_idx)

        # Fallback sanity guards
        if x_max <= x_orig + 50:
            x_max = float(min(w - 10, int(x_orig + w * 0.75)))
        if y_orig <= y_max + 50:
            y_max = float(max(10, int(y_orig - h * 0.60)))

        return {
            "x_orig": float(round(x_orig, 1)),
            "y_orig": float(round(y_orig, 1)),
            "x_max": float(round(x_max, 1)),
            "y_max": float(round(y_max, 1))
        }

    def draw_calibration_overlay(self, image_bgr: np.ndarray) -> np.ndarray:
        """Render high-contrast colored axis indicators, plot bounding box, and exact labels."""
        overlay = image_bgr.copy()
        cfg = self.config

        px_orig = int(round(cfg.x_orig))
        py_orig = int(round(cfg.y_orig))
        px_max = int(round(cfg.x_max))
        py_max = int(round(cfg.y_max))

        # 1. Semi-transparent plot region fill
        sub_overlay = overlay.copy()
        cv2.rectangle(sub_overlay, (px_orig, py_max), (px_max, py_orig), (240, 240, 200), -1)
        cv2.addWeighted(sub_overlay, 0.15, overlay, 0.85, 0, overlay)

        # 2. Plot boundary box
        cv2.rectangle(overlay, (px_orig, py_max), (px_max, py_orig), (180, 180, 180), 1)

        # 3. Y-AXIS (Survival: S=0.0 at py_orig to S=1.0 at py_max) in bright Cyan/Blue
        cv2.line(overlay, (px_orig, py_orig), (px_orig, py_max), (255, 140, 0), 3, cv2.LINE_AA)
        
        # 4. X-AXIS (Time: t_min to t_max) in bright Green
        cv2.line(overlay, (px_orig, py_orig), (px_max, py_orig), (0, 200, 50), 3, cv2.LINE_AA)

        # 5. Top reference line at S = 1.0 (dotted/thin)
        cv2.line(overlay, (px_orig, py_max), (px_max, py_max), (255, 140, 0), 1, cv2.LINE_AA)

        # 6. Origin Marker: (t=t_min, S=0.0)
        cv2.circle(overlay, (px_orig, py_orig), 7, (0, 0, 255), -1)
        cv2.circle(overlay, (px_orig, py_orig), 9, (255, 255, 255), 2)
        cv2.putText(
            overlay, f"Origin (t={cfg.t_min}, S=0.0)",
            (px_orig + 10, py_orig + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA
        )

        # 7. X-max Marker: (t=t_max, S=0.0)
        cv2.circle(overlay, (px_max, py_orig), 7, (0, 200, 50), -1)
        cv2.circle(overlay, (px_max, py_orig), 9, (255, 255, 255), 2)
        cv2.putText(
            overlay, f"X-max (t={cfg.t_max})",
            (max(px_orig + 50, px_max - 85), py_orig + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 40), 1, cv2.LINE_AA
        )

        # 8. Y-max Marker: (t=t_min, S=1.0)
        cv2.circle(overlay, (px_orig, py_max), 7, (255, 140, 0), -1)
        cv2.circle(overlay, (px_orig, py_max), 9, (255, 255, 255), 2)
        cv2.putText(
            overlay, f"Y-max (S=1.0)",
            (px_orig + 10, max(15, py_max - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 0), 1, cv2.LINE_AA
        )

        return overlay
