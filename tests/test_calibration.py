import unittest
import numpy as np
from src.models import CalibrationConfig
from src.calibration import AxisCalibrator


class TestCalibration(unittest.TestCase):

    def setUp(self):
        # Pixel coordinates: origin at (100, 400), x_max at (600, 400), y_max at (100, 100)
        # Real data: t from 0 to 50, survival from 0 to 1
        self.config = CalibrationConfig(
            x_orig=100.0,
            y_orig=400.0,
            x_max=600.0,
            y_max=100.0,
            t_min=0.0,
            t_max=50.0,
            s_min=0.0,
            s_max=1.0,
            survival_scale="0-1"
        )
        self.calibrator = AxisCalibrator(self.config)

    def test_validation(self):
        errors = self.config.validate()
        self.assertEqual(len(errors), 0)

        # Inverted axes should flag error
        bad_config = CalibrationConfig(x_orig=500.0, x_max=100.0, y_orig=100.0, y_max=400.0)
        bad_errors = bad_config.validate()
        self.assertTrue(len(bad_errors) > 0)

    def test_pixel_to_data_corners(self):
        # Origin (t_min, S_min)
        t, s = self.calibrator.pixel_to_data(100.0, 400.0)
        self.assertAlmostEqual(t, 0.0, places=4)
        self.assertAlmostEqual(s, 0.0, places=4)

        # Top-left (t_min, S_max)
        t, s = self.calibrator.pixel_to_data(100.0, 100.0)
        self.assertAlmostEqual(t, 0.0, places=4)
        self.assertAlmostEqual(s, 1.0, places=4)

        # Bottom-right (t_max, S_min)
        t, s = self.calibrator.pixel_to_data(600.0, 400.0)
        self.assertAlmostEqual(t, 50.0, places=4)
        self.assertAlmostEqual(s, 0.0, places=4)

        # Midpoint (t=25, S=0.5)
        t, s = self.calibrator.pixel_to_data(350.0, 250.0)
        self.assertAlmostEqual(t, 25.0, places=4)
        self.assertAlmostEqual(s, 0.5, places=4)

    def test_data_to_pixel_roundtrip(self):
        test_points = [(0.0, 1.0), (10.0, 0.8), (25.0, 0.5), (50.0, 0.1)]
        for orig_t, orig_s in test_points:
            px, py = self.calibrator.data_to_pixel(orig_t, orig_s)
            round_t, round_s = self.calibrator.pixel_to_data(px, py)
            self.assertAlmostEqual(orig_t, round_t, places=3)
            self.assertAlmostEqual(orig_s, round_s, places=3)

    def test_auto_detect_plot_bounds(self):
        # Create a mock white image with black axis lines
        img = np.ones((500, 700, 3), dtype=np.uint8) * 255
        # Draw axes
        img[400, 80:620] = 0  # horizontal x-axis
        img[80:400, 80] = 0   # vertical y-axis
        bounds = AxisCalibrator.auto_detect_plot_bounds(img)
        self.assertTrue(bounds["x_orig"] > 0)
        self.assertTrue(bounds["y_orig"] > bounds["y_max"])
        self.assertTrue(bounds["x_max"] > bounds["x_orig"])


if __name__ == "__main__":
    unittest.main()
