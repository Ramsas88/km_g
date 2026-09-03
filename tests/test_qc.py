import unittest
from src.models import DigitizedCurvePoint
from src.ipd_reconstruction import GuyotIPDReconstructor
from src.qc import QualityControlEngine


class TestQC(unittest.TestCase):

    def setUp(self):
        self.points = [
            DigitizedCurvePoint(1, "Arm 1", 0.0, 1.0, "0-1", 50, 50, "test"),
            DigitizedCurvePoint(1, "Arm 1", 10.0, 0.85, "0-1", 150, 80, "test"),
            DigitizedCurvePoint(1, "Arm 1", 20.0, 0.70, "0-1", 250, 110, "test"),
            DigitizedCurvePoint(1, "Arm 1", 30.0, 0.55, "0-1", 350, 140, "test"),
            DigitizedCurvePoint(1, "Arm 1", 40.0, 0.40, "0-1", 450, 170, "test"),
        ]
        self.risk_table = [
            {"time": 0.0, "n_risk": 120},
            {"time": 20.0, "n_risk": 80},
            {"time": 40.0, "n_risk": 35},
        ]

    def test_qc_evaluation_metrics(self):
        reconstructor = GuyotIPDReconstructor(
            arm_id=1,
            arm_name="Arm 1",
            curve_points=self.points,
            risk_table=self.risk_table
        )
        ipd_rows, _, _ = reconstructor.reconstruct()

        evaluation = QualityControlEngine.evaluate_arm(
            arm_id=1,
            arm_name="Arm 1",
            curve_points=self.points,
            ipd_rows=ipd_rows,
            risk_table=self.risk_table
        )

        self.assertEqual(evaluation["sample_size"], 120)
        self.assertTrue("max_absolute_error" in evaluation)
        self.assertTrue("mean_absolute_error" in evaluation)
        
        # Max absolute error between reconstructed KM and digitized points should be small (< 0.05)
        self.assertLess(evaluation["max_absolute_error"], 0.05)
        self.assertLess(evaluation["mean_absolute_error"], 0.03)

    def test_qc_full_report(self):
        reconstructor = GuyotIPDReconstructor(
            arm_id=1,
            arm_name="Arm 1",
            curve_points=self.points,
            risk_table=self.risk_table
        )
        ipd_rows, _, _ = reconstructor.reconstruct()

        report, arm_evals = QualityControlEngine.generate_full_report(
            arm_curves={1: self.points},
            arm_ipds={1: ipd_rows},
            arm_risks={1: self.risk_table},
            arm_names={1: "Arm 1"}
        )

        self.assertIn("Arm 1", report.extracted_points_per_arm)
        self.assertIn("Arm 1", report.reconstructed_sample_size_per_arm)
        self.assertEqual(report.reconstructed_sample_size_per_arm["Arm 1"], 120)
        report_dict = report.to_dict()
        self.assertIn("created_at", report_dict)


if __name__ == "__main__":
    unittest.main()
