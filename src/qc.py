"""Quality control evaluation, lifelines KM curve recalculation, and error benchmarking."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import io
from src.models import DigitizedCurvePoint, PseudoIPDRow, QCReport


class QualityControlEngine:
    """Evaluates pseudo-IPD fidelity against digitized survival curves."""

    @staticmethod
    def evaluate_arm(
        arm_id: int,
        arm_name: str,
        curve_points: List[DigitizedCurvePoint],
        ipd_rows: List[PseudoIPDRow],
        risk_table: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Recalculates KM estimates from IPD and evaluates absolute errors against digitized curve.
        """
        if not ipd_rows:
            return {
                "arm_id": arm_id,
                "arm_name": arm_name,
                "sample_size": 0,
                "events": 0,
                "censored": 0,
                "max_absolute_error": 1.0,
                "mean_absolute_error": 1.0,
                "evaluated_points": 0,
                "warnings": ["No IPD rows available to evaluate."]
            }

        ipd_df = pd.DataFrame([r.to_dict() for r in ipd_rows])
        kmf = KaplanMeierFitter()
        kmf.fit(durations=ipd_df["time"], event_observed=ipd_df["event"], label=arm_name)

        # Compare at digitized time points
        sorted_pts = sorted(curve_points, key=lambda p: p.time)
        eval_times = [p.time for p in sorted_pts]
        eval_surv_dig = [p.survival for p in sorted_pts]

        # Normalization if 0-100 scale
        if max(eval_surv_dig) > 1.5:
            eval_surv_dig = [s / 100.0 for s in eval_surv_dig]

        # Query lifelines KM estimate at each digitized time point
        reconstructed_survs = kmf.predict(eval_times).values

        abs_errors = np.abs(np.array(eval_surv_dig) - reconstructed_survs)
        max_ae = float(np.max(abs_errors)) if len(abs_errors) > 0 else 0.0
        mean_ae = float(np.mean(abs_errors)) if len(abs_errors) > 0 else 0.0

        events_count = int(ipd_df["event"].sum())
        censored_count = int(len(ipd_df) - events_count)

        arm_warnings = []
        if max_ae > 0.08:
            arm_warnings.append(
                f"High maximum absolute error ({max_ae:.3f}) between reconstructed and digitized KM curves for arm '{arm_name}'."
            )

        return {
            "arm_id": arm_id,
            "arm_name": arm_name,
            "sample_size": len(ipd_df),
            "events": events_count,
            "censored": censored_count,
            "max_absolute_error": round(max_ae, 4),
            "mean_absolute_error": round(mean_ae, 4),
            "evaluated_points": len(eval_times),
            "kmf": kmf,
            "comparison_df": pd.DataFrame({
                "time": eval_times,
                "digitized_survival": np.round(eval_surv_dig, 4),
                "reconstructed_survival": np.round(reconstructed_survs, 4),
                "absolute_error": np.round(abs_errors, 4)
            }),
            "warnings": arm_warnings
        }

    @classmethod
    def generate_full_report(
        cls,
        arm_curves: Dict[int, List[DigitizedCurvePoint]],
        arm_ipds: Dict[int, List[PseudoIPDRow]],
        arm_risks: Dict[int, List[Dict[str, Any]]],
        arm_names: Dict[int, str]
    ) -> Tuple[QCReport, Dict[int, Dict[str, Any]]]:
        """Generates a comprehensive QCReport across all arms."""
        report = QCReport()
        arm_evals = {}

        for arm_id, points in arm_curves.items():
            name = arm_names.get(arm_id, f"Arm {arm_id}")
            ipd = arm_ipds.get(arm_id, [])
            risk = arm_risks.get(arm_id, [])

            evaluation = cls.evaluate_arm(arm_id, name, points, ipd, risk)
            arm_evals[arm_id] = evaluation

            report.extracted_points_per_arm[name] = len(points)
            report.number_at_risk_used[name] = risk
            report.estimated_events_per_arm[name] = evaluation["events"]
            report.estimated_censored_per_arm[name] = evaluation["censored"]
            report.reconstructed_sample_size_per_arm[name] = evaluation["sample_size"]
            report.max_absolute_km_error[name] = evaluation["max_absolute_error"]
            report.mean_absolute_km_error[name] = evaluation["mean_absolute_error"]

            if evaluation["warnings"]:
                report.warnings.extend(evaluation["warnings"])

        return report, arm_evals

    @staticmethod
    def plot_reconstructed_vs_digitized(
        arm_evals: Dict[int, Dict[str, Any]],
        arm_curves: Dict[int, List[DigitizedCurvePoint]]
    ) -> plt.Figure:
        """Create high-quality comparative matplotlib plot showing digitized vs reconstructed curves."""
        fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for idx, (arm_id, ev) in enumerate(arm_evals.items()):
            color = colors[idx % len(colors)]
            arm_name = ev["arm_name"]

            # Plot digitized curve points
            pts = sorted(arm_curves.get(arm_id, []), key=lambda p: p.time)
            if pts:
                t_dig = [p.time for p in pts]
                s_dig = [p.survival if p.survival <= 1.0 else p.survival / 100.0 for p in pts]
                ax.step(t_dig, s_dig, where="post", color=color, linestyle="--", alpha=0.65,
                        label=f"{arm_name} (Digitized)")

            # Plot reconstructed lifelines KM curve
            kmf = ev.get("kmf")
            if kmf is not None:
                kmf.plot_survival_function(ax=ax, color=color, linewidth=2,
                                           label=f"{arm_name} (Reconstructed IPD, N={ev['sample_size']}, d={ev['events']})")

        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Kaplan-Meier QC: Digitized Curves vs Reconstructed Pseudo-IPD")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        return fig
