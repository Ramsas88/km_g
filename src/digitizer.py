"""Image processing, color segmentation, multi-track line tracing, and stepwise KM curve extraction."""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.models import CalibrationConfig, DigitizedCurvePoint
from src.calibration import AxisCalibrator


class KMDigitizer:
    """Extracts and cleans Kaplan-Meier curves from calibrated plot images."""

    def __init__(self, image_bgr: np.ndarray, calibrator: AxisCalibrator, source_name: str = "km_plot"):
        self.image = image_bgr
        self.calibrator = calibrator
        self.config = calibrator.config
        self.source_name = source_name

    def is_colored_image(self) -> bool:
        """Determines whether the plot contains colored curves or is monochrome / grayscale."""
        if self.image is None:
            return False
        cfg = self.config
        x1, x2 = int(max(0, cfg.x_orig)), int(min(self.image.shape[1], cfg.x_max))
        y1, y2 = int(max(0, cfg.y_max)), int(min(self.image.shape[0], cfg.y_orig))
        if x2 <= x1 or y2 <= y1:
            return False

        roi = self.image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        # Colored pixels have non-trivial saturation and brightness
        colored_pixels = np.sum((sat > 38) & (val > 35) & (val < 245))
        return bool(colored_pixels > 600)

    def detect_curve_colors(self, max_colors: int = 4) -> List[Dict[str, Any]]:
        """
        Identify prominent curve colors within the plot region.
        Excludes white/near-white background and pure black grid/axis lines.
        Returns candidate arm colors with hex, RGB, and BGR values.
        """
        cfg = self.config
        x1, x2 = int(max(0, cfg.x_orig)), int(min(self.image.shape[1], cfg.x_max))
        y1, y2 = int(max(0, cfg.y_max)), int(min(self.image.shape[0], cfg.y_orig))

        if x2 <= x1 or y2 <= y1:
            return [{"arm_id": 1, "arm_name": "Arm 1", "color_bgr": [255, 0, 0], "hex": "#0000ff"}]

        roi = self.image[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Saturated colored pixels (S > 35, V > 35 and V < 240)
        sat_mask = (hsv_roi[:, :, 1] > 35) & (hsv_roi[:, :, 2] > 35) & (hsv_roi[:, :, 2] < 240)
        valid_pixels = roi[sat_mask]

        candidate_arms = []
        if len(valid_pixels) >= 100:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            k = min(max_colors, max(1, len(valid_pixels) // 200))
            _, labels, centers = cv2.kmeans(
                valid_pixels.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )

            unique, counts = np.unique(labels, return_counts=True)
            sorted_clusters = sorted(zip(counts, centers), key=lambda x: x[0], reverse=True)

            for idx, (cnt, center) in enumerate(sorted_clusters[:max_colors]):
                b, g, r = int(center[0]), int(center[1]), int(center[2])
                hex_code = f"#{r:02x}{g:02x}{b:02x}"
                candidate_arms.append({
                    "arm_id": idx + 1,
                    "arm_name": f"Arm {idx + 1}",
                    "color_bgr": [b, g, r],
                    "hex": hex_code,
                    "pixel_count": int(cnt)
                })

        # Fallback if image is monochrome or low-saturation
        if not candidate_arms:
            candidate_arms = [
                {"arm_id": 1, "arm_name": "Upper Arm (Arm 1)", "color_bgr": [20, 20, 20], "hex": "#141414", "pixel_count": 0},
                {"arm_id": 2, "arm_name": "Lower Arm (Arm 2)", "color_bgr": [80, 80, 80], "hex": "#505050", "pixel_count": 0}
            ]

        return candidate_arms

    def extract_curve_by_color(
        self, 
        target_bgr: List[int], 
        arm_id: int, 
        arm_name: str, 
        color_tol: int = 55,
        dash_bridge_len: int = 12
    ) -> List[DigitizedCurvePoint]:
        """
        Extracts a single survival curve matching the target BGR color within the plot boundaries.
        Uses color distance in LAB color space and morphological closing to bridge gaps/ticks.
        """
        cfg = self.config
        x_start = int(round(cfg.x_orig)) + 2
        x_end = int(round(cfg.x_max)) - 2
        y_top = int(round(cfg.y_max)) + 2
        y_bottom = int(round(cfg.y_orig)) - 2

        if x_end <= x_start or y_bottom <= y_top:
            return []

        # Convert image and target to LAB color space for perceptual color uniformity
        target_np = np.uint8([[target_bgr]])
        target_lab = cv2.cvtColor(target_np, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        image_lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Delta E color distance
        diff = np.linalg.norm(image_lab - target_lab, axis=2)
        mask = (diff <= color_tol).astype(np.uint8) * 255

        # Restrict strictly inside plot area
        plot_mask = np.zeros_like(mask)
        plot_mask[y_top:y_bottom, x_start:x_end] = 255
        line_mask = cv2.bitwise_and(mask, plot_mask)

        # Morphological closing to bridge dashes and anti-aliased breaks
        if dash_bridge_len > 1:
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dash_bridge_len, 2))
            line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, bridge_kernel)

        raw_points = []
        detected_x_count = 0
        total_x_span = x_end - x_start
        last_py = float(y_top)

        for px in range(x_start, x_end + 1):
            column = line_mask[y_top:y_bottom, px]
            ys = np.where(column > 0)[0]
            if len(ys) > 0:
                cand_ys = ys + y_top
                # Survival curves drop downward: prioritize candidate ys that do not jump upwards
                valid_cand = [y for y in cand_ys if y >= last_py - 4]
                if valid_cand:
                    best_py = float(np.median(valid_cand))
                else:
                    best_py = float(np.median(cand_ys))

                last_py = max(last_py, best_py)
                raw_points.append((px, last_py, True))
                detected_x_count += 1
            else:
                # Gap: hold previous horizontal level
                raw_points.append((px, last_py, False))

        coverage = detected_x_count / max(1, total_x_span)
        confidence = float(min(1.0, max(0.15, coverage)))

        return self.clean_and_stepify_points(
            raw_points, arm_id, arm_name, confidence, "color_segmentation"
        )

    def extract_multitrack_curves(
        self,
        num_tracks: int = 2,
        darkness_thresh: int = 175,
        dash_bridge_len: int = 16,
        arm_names: Optional[List[str]] = None
    ) -> Dict[int, List[DigitizedCurvePoint]]:
        """
        Extracts multiple survival curves from monochrome, grayscale, or solid vs dashed plots.
        Separates curves into distinct trajectories (e.g. Upper Arm, Lower Arm).
        """
        cfg = self.config
        x_start = int(round(cfg.x_orig)) + 3
        x_end = int(round(cfg.x_max)) - 3
        y_top = int(round(cfg.y_max)) + 3
        y_bottom = int(round(cfg.y_orig)) - 4

        if x_end <= x_start or y_bottom <= y_top:
            return {}

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Binary line mask
        bin_mask = ((gray < darkness_thresh) & (gray > 10)).astype(np.uint8) * 255
        roi_mask = np.zeros_like(bin_mask)
        roi_mask[y_top:y_bottom, x_start:x_end] = 255
        lines_only = cv2.bitwise_and(bin_mask, roi_mask)

        # Exclude legend text blocks (small rectangular boxes located away from axes)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lines_only)
        clean_lines = lines_only.copy()
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            # Legend text boxes typically occupy a bounded area in upper-right or middle
            if x > (x_start + (x_end - x_start) * 0.55) and y < (y_top + (y_bottom - y_top) * 0.7):
                if 15 < w < 140 and 8 < h < 45 and 60 < area < 2500:
                    clean_lines[labels == i] = 0

        # Horizontally bridge dashed lines
        if dash_bridge_len > 1:
            dash_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dash_bridge_len, 2))
            bridged = cv2.morphologyEx(clean_lines, cv2.MORPH_CLOSE, dash_kernel)
        else:
            bridged = clean_lines

        # Prepare tracking trajectories
        tracks_raw = [[] for _ in range(num_tracks)]
        curr_ys = [float(y_top) for _ in range(num_tracks)]

        for px in range(x_start, x_end + 1):
            col_ys = np.where(bridged[y_top:y_bottom, px] > 0)[0] + y_top
            if len(col_ys) == 0:
                for tr in range(num_tracks):
                    tracks_raw[tr].append((px, curr_ys[tr], False))
                continue

            # Cluster vertical pixels in this column into discrete line centers
            clusters = []
            curr_clust = [col_ys[0]]
            for y in col_ys[1:]:
                if y - curr_clust[-1] <= 6:
                    curr_clust.append(y)
                else:
                    clusters.append(float(np.median(curr_clust)))
                    curr_clust = [y]
            if curr_clust:
                clusters.append(float(np.median(curr_clust)))

            clusters.sort()  # Top-to-bottom: index 0 is topmost line (highest survival)

            if len(clusters) == 1:
                c = clusters[0]
                # If near beginning, initialize all tracks together
                if px < x_start + 12:
                    for tr in range(num_tracks):
                        curr_ys[tr] = max(curr_ys[tr], c)
                        tracks_raw[tr].append((px, curr_ys[tr], True))
                else:
                    # Assign to track with minimum distance
                    best_tr = int(np.argmin([abs(c - curr_ys[tr]) for tr in range(num_tracks)]))
                    curr_ys[best_tr] = max(curr_ys[best_tr], c)
                    for tr in range(num_tracks):
                        tracks_raw[tr].append((px, curr_ys[tr], tr == best_tr))
            else:
                # Multiple line candidates
                if num_tracks == 2:
                    c_top = clusters[0]
                    c_bot = clusters[-1]
                    if c_top >= curr_ys[0] - 4:
                        curr_ys[0] = max(curr_ys[0], c_top)
                    if c_bot >= curr_ys[1] - 4:
                        curr_ys[1] = max(curr_ys[1], c_bot)
                    # Enforce lower track does not cross above upper track
                    if curr_ys[1] < curr_ys[0]:
                        curr_ys[1] = curr_ys[0]
                    tracks_raw[0].append((px, curr_ys[0], True))
                    tracks_raw[1].append((px, curr_ys[1], True))
                else:
                    # Generic K tracks: partition clusters across tracks
                    step = max(1, len(clusters) // num_tracks)
                    for tr in range(num_tracks):
                        cand = clusters[min(len(clusters) - 1, tr * step)]
                        if cand >= curr_ys[tr] - 4:
                            curr_ys[tr] = max(curr_ys[tr], cand)
                        tracks_raw[tr].append((px, curr_ys[tr], True))

        # Convert to DigitizedCurvePoint objects
        results: Dict[int, List[DigitizedCurvePoint]] = {}
        for tr in range(num_tracks):
            arm_id = tr + 1
            if arm_names and tr < len(arm_names):
                name = arm_names[tr]
            else:
                name = f"Upper Arm (Arm {arm_id})" if tr == 0 else f"Lower Arm (Arm {arm_id})"

            raw_pts = tracks_raw[tr]
            detected_cnt = sum(1 for _, _, det in raw_pts if det)
            conf = float(min(1.0, max(0.2, detected_cnt / max(1, len(raw_pts)))))
            pts = self.clean_and_stepify_points(raw_pts, arm_id, name, conf, "multi_track_tracer")
            results[arm_id] = pts

        return results

    def clean_and_stepify_points(
        self,
        raw_px_py_list: List[Tuple[int, float, bool]],
        arm_id: int,
        arm_name: str,
        confidence: float,
        extraction_method: str = "auto_detection"
    ) -> List[DigitizedCurvePoint]:
        """Converts pixel coordinates to real-world (time, survival) points with step preservation."""
        cfg = self.config
        cleaned: List[DigitizedCurvePoint] = []

        # Start point at t_min, S=1.0
        p0_x, p0_y = self.calibrator.data_to_pixel(cfg.t_min, 1.0)
        cleaned.append(DigitizedCurvePoint(
            arm_id=arm_id,
            arm_name=arm_name,
            time=float(cfg.t_min),
            survival=1.0,
            survival_scale=cfg.survival_scale,
            pixel_x=float(p0_x),
            pixel_y=float(p0_y),
            source_image=self.source_name,
            extraction_method=extraction_method,
            confidence=confidence
        ))

        current_min_s = 1.0
        last_added_t = cfg.t_min
        last_added_s = 1.0

        for px, py, is_detected in raw_px_py_list:
            t, s = self.calibrator.pixel_to_data(px, py)
            
            # Non-increasing survival
            if s > current_min_s:
                s = current_min_s
            else:
                current_min_s = s

            s_diff = abs(s - last_added_s)
            t_diff = abs(t - last_added_t)

            # Preserve vertical drop points and regular interval points
            if s_diff >= 0.004 or t_diff >= (cfg.t_max - cfg.t_min) / 45.0:
                if s_diff >= 0.004 and t_diff > 0.01:
                    cleaned.append(DigitizedCurvePoint(
                        arm_id=arm_id,
                        arm_name=arm_name,
                        time=float(round(t, 3)),
                        survival=float(round(last_added_s, 4)),
                        survival_scale=cfg.survival_scale,
                        pixel_x=float(round(px, 1)),
                        pixel_y=float(round(self.calibrator.data_to_pixel(t, last_added_s)[1], 1)),
                        source_image=self.source_name,
                        extraction_method=extraction_method,
                        confidence=confidence
                    ))

                cleaned.append(DigitizedCurvePoint(
                    arm_id=arm_id,
                    arm_name=arm_name,
                    time=float(round(t, 3)),
                    survival=float(round(s, 4)),
                    survival_scale=cfg.survival_scale,
                    pixel_x=float(round(px, 1)),
                    pixel_y=float(round(py, 1)),
                    source_image=self.source_name,
                    extraction_method=extraction_method,
                    confidence=confidence
                ))
                last_added_t = t
                last_added_s = s

        # Ensure curve terminates at t_max
        if last_added_t < cfg.t_max - 1e-4:
            p_end_x, p_end_y = self.calibrator.data_to_pixel(cfg.t_max, last_added_s)
            cleaned.append(DigitizedCurvePoint(
                arm_id=arm_id,
                arm_name=arm_name,
                time=float(round(cfg.t_max, 3)),
                survival=float(round(last_added_s, 4)),
                survival_scale=cfg.survival_scale,
                pixel_x=float(round(p_end_x, 1)),
                pixel_y=float(round(p_end_y, 1)),
                source_image=self.source_name,
                extraction_method=extraction_method,
                confidence=confidence
            ))

        return cleaned

    def draw_curve_overlay_on_image(
        self,
        arm_curves: Dict[int, List[DigitizedCurvePoint]],
        arm_colors: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """
        Renders the extracted step curves and digitized points directly over a copy of the source image.
        Allows instant visual verification of curve tracking against published lines.
        """
        overlay = self.image.copy()
        default_colors = [(255, 100, 0), (0, 165, 255), (0, 255, 120), (255, 0, 220)]

        for idx, (arm_id, pts) in enumerate(arm_curves.items()):
            if not pts:
                continue
            color = default_colors[idx % len(default_colors)]
            if arm_colors and arm_id in arm_colors:
                # Convert hex to BGR
                h = arm_colors[arm_id].lstrip("#")
                if len(h) == 6:
                    color = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

            # Draw step function line connecting consecutive points
            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i + 1]
                px1, py1 = int(round(p1.pixel_x)), int(round(p1.pixel_y))
                px2, py2 = int(round(p2.pixel_x)), int(round(p2.pixel_y))

                # Step function: horizontal segment then vertical drop
                cv2.line(overlay, (px1, py1), (px2, py1), color, 2, cv2.LINE_AA)
                cv2.line(overlay, (px2, py1), (px2, py2), color, 2, cv2.LINE_AA)
                # Small circle on point
                cv2.circle(overlay, (px2, py2), 3, color, -1)

        return overlay

    @staticmethod
    def enforce_monotonicity(points: List[DigitizedCurvePoint]) -> List[DigitizedCurvePoint]:
        """Ensures survival values strictly do not increase over time."""
        if not points:
            return []
        sorted_pts = sorted(points, key=lambda p: p.time)
        cleaned = []
        curr_min = 1.0
        for p in sorted_pts:
            curr_min = min(curr_min, p.survival)
            cleaned.append(DigitizedCurvePoint(
                arm_id=p.arm_id,
                arm_name=p.arm_name,
                time=p.time,
                survival=curr_min,
                survival_scale=p.survival_scale,
                pixel_x=p.pixel_x,
                pixel_y=p.pixel_y,
                source_image=p.source_image,
                extraction_method=p.extraction_method,
                confidence=p.confidence
            ))
        return cleaned

    @staticmethod
    def simplify_curve(points: List[DigitizedCurvePoint], tolerance: float = 0.01) -> List[DigitizedCurvePoint]:
        """Reduces redundant points along flat horizontal survival segments."""
        if len(points) <= 2:
            return points
        simplified = [points[0]]
        for i in range(1, len(points) - 1):
            prev_s = simplified[-1].survival
            curr_s = points[i].survival
            next_s = points[i + 1].survival
            if abs(curr_s - prev_s) < tolerance and abs(next_s - curr_s) < tolerance:
                continue
            simplified.append(points[i])
        simplified.append(points[-1])
        return simplified

    @staticmethod
    def shift_curve(points: List[DigitizedCurvePoint], delta_s: float) -> List[DigitizedCurvePoint]:
        """Shifts all survival points by delta_s and clamps within [0.0, 1.0]."""
        shifted = []
        for p in points:
            new_s = max(0.0, min(1.0, p.survival + delta_s))
            shifted.append(DigitizedCurvePoint(
                arm_id=p.arm_id,
                arm_name=p.arm_name,
                time=p.time,
                survival=float(round(new_s, 4)),
                survival_scale=p.survival_scale,
                pixel_x=p.pixel_x,
                pixel_y=p.pixel_y,
                source_image=p.source_image,
                extraction_method=p.extraction_method,
                confidence=p.confidence
            ))
        return KMDigitizer.enforce_monotonicity(shifted)
