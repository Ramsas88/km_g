import unittest
import numpy as np
from src.models import DigitizedCurvePoint
from src.ipd_reconstruction import GuyotIPDReconstructor


class TestIPDReconstruction(unittest.TestCase):

    def setUp(self):
        # Known synthetic step survival curve:
        # t=0 -> S=1.0
        # t=10 -> S=0.8
        # t=20 -> S=0.6
        # t=30 -> S=0.4
        # t=40 -> S=0.2
        self.points = [
            DigitizedCurvePoint(1, "Arm 1", 0.0, 1.0, "0-1", 50, 50, "test"),
            DigitizedCurvePoint(1, "Arm 1", 10.0, 0.8, "0-1", 150, 90, "test"),
            DigitizedCurvePoint(1, "Arm 1", 20.0, 0.6, "0-1", 250, 130, "test"),
            DigitizedCurvePoint(1, "Arm 1", 30.0, 0.4, "0-1", 350, 170, "test"),
            DigitizedCurvePoint(1, "Arm 1", 40.0, 0.2, "0-1", 450, 210, "test"),
        ]

        # Published risk table:
        # t=0: 100 at risk
        # t=20: 55 at risk (some events + some censored)
        # t=40: 15 at risk
        self.risk_table = [
            {"time": 0.0, "n_risk": 100},
            {"time": 20.0, "n_risk": 55},
            {"time": 40.0, "n_risk": 15},
        ]

    def test_reconstruct_sample_size_and_events(self):
        reconstructor = GuyotIPDReconstructor(
            arm_id=1,
            arm_name="Arm 1",
            curve_points=self.points,
            risk_table=self.risk_table
        )
        ipd_rows, summary_df, meta = reconstructor.reconstruct()

        # Reconstructed sample size must equal n_risk at t=0
        self.assertEqual(len(ipd_rows), 100)
        self.assertEqual(meta["reconstructed_sample_size"], 100)

        # Check that events (1) and censors (0) sum to 100
        events = [r for r in ipd_rows if r.event == 1]
        censored = [r for r in ipd_rows if r.event == 0]
        self.assertEqual(len(events) + len(censored), 100)

        # Both events and censors should be positive in this scenario
        self.assertTrue(len(events) > 0)
        self.assertTrue(len(censored) > 0)

        # Check schema of ipd_rows
        sample_row = ipd_rows[0]
        self.assertTrue(sample_row.patient_id.startswith("arm1_pat_"))
        self.assertEqual(sample_row.arm_id, 1)
        self.assertEqual(sample_row.arm_name, "Arm 1")
        self.assertIn(sample_row.event, [0, 1])

    def test_total_events_constraint(self):
        # Enforce that total events is constrained to 55 (unconstrained was 50)
        target_events = 55
        reconstructor = GuyotIPDReconstructor(
            arm_id=1,
            arm_name="Arm 1",
            curve_points=self.points,
            risk_table=self.risk_table,
            tot_events=target_events
        )
        ipd_rows, summary_df, meta = reconstructor.reconstruct()

        self.assertEqual(len(ipd_rows), 100)
        actual_events = sum(1 for r in ipd_rows if r.event == 1)
        self.assertEqual(actual_events, target_events)

    def test_single_interval_risk(self):
        # Test edge case: single risk table entry at t=0
        single_risk = [{"time": 0.0, "n_risk": 50}]
        reconstructor = GuyotIPDReconstructor(
            arm_id=1,
            arm_name="Arm 1",
            curve_points=self.points,
            risk_table=single_risk
        )
        ipd_rows, summary_df, meta = reconstructor.reconstruct()
        self.assertEqual(len(ipd_rows), 50)


if __name__ == "__main__":
    unittest.main()
