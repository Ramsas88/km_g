import unittest
import numpy as np
import cv2
from src.models import CalibrationConfig, DigitizedCurvePoint
from src.calibration import AxisCalibrator
from src.digitizer import KMDigitizer


class TestDigitizer(unittest.TestCase):

    def setUp(self):
        self.config = CalibrationConfig(
            x_orig=50.0,
            y_orig=250.0,
            x_max=450.0,
            y_max=50.0,
            t_min=0.0,
            t_max=40.0,
            s_min=0.0,
            s_max=1.0
        )
        self.calibrator = AxisCalibrator(self.config)

    def test_enforce_monotonicity(self):
        points = [
            DigitizedCurvePoint(1, "Arm 1", 0.0, 1.0, "0-1", 50, 50, "test"),
            DigitizedCurvePoint(1, "Arm 1", 10.0, 0.8, "0-1", 150, 90, "test"),
            DigitizedCurvePoint(1, "Arm 1", 20.0, 0.85, "0-1", 250, 80, "test"),  # Non-monotonic bump
            DigitizedCurvePoint(1, "Arm 1", 30.0, 0.6, "0-1", 350, 130, "test"),
        ]
        cleaned = KMDigitizer.enforce_monotonicity(points)
        self.assertEqual(len(cleaned), 4)
        self.assertEqual(cleaned[0].survival, 1.0)
        self.assertEqual(cleaned[1].survival, 0.8)
        self.assertEqual(cleaned[2].survival, 0.8)  # Bump corrected down
        self.assertEqual(cleaned[3].survival, 0.6)

    def test_simplify_curve(self):
        points = [
            DigitizedCurvePoint(1, "Arm 1", 0.0, 1.0, "0-1", 50, 50, "test"),
            DigitizedCurvePoint(1, "Arm 1", 2.0, 1.0, "0-1", 70, 50, "test"),
            DigitizedCurvePoint(1, "Arm 1", 4.0, 1.0, "0-1", 90, 50, "test"),
            DigitizedCurvePoint(1, "Arm 1", 10.0, 0.7, "0-1", 150, 110, "test"),
            DigitizedCurvePoint(1, "Arm 1", 20.0, 0.7, "0-1", 250, 110, "test"),
            DigitizedCurvePoint(1, "Arm 1", 30.0, 0.4, "0-1", 350, 170, "test"),
        ]
        simplified = KMDigitizer.simplify_curve(points, tolerance=0.01)
        self.assertTrue(len(simplified) < len(points))
        self.assertEqual(simplified[0].time, 0.0)
        self.assertEqual(simplified[-1].time, 30.0)

    def test_shift_curve(self):
        points = [
            DigitizedCurvePoint(1, "Arm 1", 0.0, 1.0, "0-1", 50, 50, "test"),
            DigitizedCurvePoint(1, "Arm 1", 10.0, 0.7, "0-1", 150, 110, "test"),
        ]
        shifted = KMDigitizer.shift_curve(points, delta_s=-0.1)
        self.assertAlmostEqual(shifted[0].survival, 0.9, places=3)
        self.assertAlmostEqual(shifted[1].survival, 0.6, places=3)

    def test_synthetic_curve_extraction(self):
        # Create a synthetic image with a blue step curve
        img = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.line(img, (50, 50), (200, 50), (255, 0, 0), 2)
        cv2.line(img, (200, 50), (200, 150), (255, 0, 0), 2)
        cv2.line(img, (200, 150), (450, 150), (255, 0, 0), 2)

        digitizer = KMDigitizer(img, self.calibrator, source_name="synthetic")
        curve_pts = digitizer.extract_curve_by_color(
            target_bgr=[255, 0, 0], arm_id=1, arm_name="Blue Arm", color_tol=50
        )

        self.assertTrue(len(curve_pts) >= 2)
        self.assertAlmostEqual(curve_pts[0].survival, 1.0, places=2)
        self.assertAlmostEqual(curve_pts[-1].survival, 0.5, delta=0.08)

    def test_multitrack_extraction(self):
        # Create a synthetic monochrome image with 2 black curves
        img = np.ones((300, 500, 3), dtype=np.uint8) * 255
        # Upper curve
        cv2.line(img, (50, 50), (250, 50), (0, 0, 0), 2)
        cv2.line(img, (250, 50), (250, 120), (0, 0, 0), 2)
        cv2.line(img, (250, 120), (450, 120), (0, 0, 0), 2)
        # Lower curve (dashed or lower level)
        cv2.line(img, (50, 50), (150, 50), (0, 0, 0), 2)
        cv2.line(img, (150, 50), (150, 180), (0, 0, 0), 2)
        cv2.line(img, (150, 180), (450, 180), (0, 0, 0), 2)

        digitizer = KMDigitizer(img, self.calibrator, source_name="synthetic_mono")
        tracks = digitizer.extract_multitrack_curves(num_tracks=2)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(1 in tracks and 2 in tracks)
        # Upper track (track 1) should have higher or equal survival than lower track (track 2)
        s_upper = tracks[1][-1].survival
        s_lower = tracks[2][-1].survival
        self.assertTrue(s_upper >= s_lower)


if __name__ == "__main__":
    unittest.main()
