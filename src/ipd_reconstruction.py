"""Reconstructs pseudo-Individual Patient Data (pseudo-IPD) using the Guyot et al. (2012) algorithm."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
from src.models import DigitizedCurvePoint, RiskInterval, PseudoIPDRow


class GuyotIPDReconstructor:
    """
    Python implementation of the Guyot et al. (2012) algorithm:
    'Enhanced secondary analysis of survival data: reconstructing the data from published Kaplan-Meier survival curves'
    Adapted directly from km_algoritham.pdf.
    """

    def __init__(
        self,
        arm_id: int,
        arm_name: str,
        curve_points: List[DigitizedCurvePoint],
        risk_table: List[Dict[str, Any]],
        tot_events: Optional[int] = None
    ):
        self.arm_id = arm_id
        self.arm_name = arm_name
        self.raw_curve_points = curve_points
        self.risk_table = risk_table
        self.tot_events = tot_events
        self.warnings: List[str] = []

    def prepare_curve_and_risk(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Validates, cleans, and aligns curve points and risk intervals.
        Returns:
          t_S: 1D array of survival times (starts with 0.0, S=1.0)
          S: 1D array of survival probabilities (0.0 to 1.0)
          t_risk: 1D array of published risk times
          n_risk: 1D array of published number at risk
          lower: 1D array of lower index in t_S for each interval (0-indexed)
          upper: 1D array of upper index in t_S for each interval (0-indexed)
        """
        if not self.raw_curve_points:
            raise ValueError(f"Arm '{self.arm_name}': No digitized curve points provided.")

        # Sort curve points by time
        sorted_pts = sorted(self.raw_curve_points, key=lambda p: p.time)
        times = [p.time for p in sorted_pts]
        survs = [p.survival for p in sorted_pts]

        # Normalize survival scale if necessary
        if max(survs) > 1.5:
            survs = [s / 100.0 for s in survs]
        survs = [max(0.0, min(1.0, s)) for s in survs]

        # Enforce non-increasing survival
        for idx in range(1, len(survs)):
            if survs[idx] > survs[idx - 1]:
                survs[idx] = survs[idx - 1]

        # Ensure t=0, S=1.0 exists
        if times[0] > 1e-4:
            times.insert(0, 0.0)
            survs.insert(0, 1.0)
        else:
            survs[0] = 1.0

        t_S = np.array(times, dtype=np.float64)
        S = np.array(survs, dtype=np.float64)

        # Parse risk table
        if not self.risk_table:
            raise ValueError(f"Arm '{self.arm_name}': Published number-at-risk table is required for IPD reconstruction.")

        # Sort risk table by time
        sorted_risk = sorted(self.risk_table, key=lambda r: float(r["time"]))
        t_risk = np.array([float(r["time"]) for r in sorted_risk], dtype=np.float64)
        n_risk = np.array([int(r["n_risk"]) for r in sorted_risk], dtype=np.int64)

        n_int = len(t_risk)
        if n_int < 1:
            raise ValueError("Number-at-risk table must have at least one entry.")

        # Validate that n_risk is non-increasing
        for i in range(1, n_int):
            if n_risk[i] > n_risk[i - 1]:
                self.warnings.append(
                    f"Warning: Published number at risk increased from {n_risk[i-1]} at t={t_risk[i-1]} to {n_risk[i]} at t={t_risk[i]}. Clamped to {n_risk[i-1]}."
                )
                n_risk[i] = n_risk[i - 1]

        # Map t_risk to lower indices in t_S
        lower = np.zeros(n_int, dtype=np.int64)
        for i in range(n_int):
            t_curr = t_risk[i]
            # Find first index where t_S >= t_curr
            matches = np.where(t_S >= t_curr - 1e-6)[0]
            if len(matches) > 0:
                lower[i] = matches[0]
            else:
                lower[i] = len(t_S) - 1

        # Enforce lower indices are strictly increasing
        for i in range(1, n_int):
            if lower[i] <= lower[i - 1]:
                # Need to insert an artificial point if curve points were too sparse
                lower[i] = min(len(t_S) - 1, lower[i - 1] + 1)

        # Calculate upper indices
        upper = np.zeros(n_int, dtype=np.int64)
        for i in range(n_int - 1):
            upper[i] = max(lower[i], lower[i + 1] - 1)
        upper[n_int - 1] = len(t_S) - 1

        return t_S, S, t_risk, n_risk, lower, upper

    def reconstruct(self) -> Tuple[List[PseudoIPDRow], pd.DataFrame, Dict[str, Any]]:
        """
        Executes the Guyot reconstruction algorithm.
        Returns:
          ipd_rows: List of PseudoIPDRow objects
          summary_df: DataFrame with interval-level calculations (t, n_hat, d, cen)
          metadata: Dictionary with reconstruction metrics
        """
        t_S, S, t_risk, n_risk, lower, upper = self.prepare_curve_and_risk()

        n_int = len(n_risk)
        n_t = len(t_S)

        # Initialize tracking vectors
        n_censor = np.zeros(n_int, dtype=np.float64)
        n_hat = np.zeros(n_t + 1, dtype=np.float64)
        n_hat[:] = n_risk[0] + 1
        cen = np.zeros(n_t, dtype=np.int64)
        d = np.zeros(n_t, dtype=np.int64)
        KM_hat = np.ones(n_t, dtype=np.float64)
        last_i = np.zeros(n_int, dtype=np.int64)

        if n_int > 1:
            for i in range(n_int - 1):
                low_curr = lower[i]
                low_next = lower[i + 1]
                up_curr = upper[i]

                s_low_next = S[low_next]
                s_low_curr = max(1e-8, S[low_curr])
                
                # First approximation of number censored in interval i
                approx_cen = n_risk[i] * (s_low_next / s_low_curr) - n_risk[i + 1]
                n_censor[i] = round(approx_cen)

                # Iteratively adjust n_censor[i] until n_hat matches n_risk[i+1]
                iter_count = 0
                max_iter = 500

                while iter_count < max_iter:
                    iter_count += 1

                    if n_censor[i] <= 0:
                        cen[low_curr : up_curr + 1] = 0
                        n_censor[i] = 0
                    else:
                        num_c = int(round(n_censor[i]))
                        # Distribute censored observations evenly over interval
                        j_vals = np.arange(1, num_c + 1)
                        cen_t = t_S[low_curr] + j_vals * (t_S[low_next] - t_S[low_curr]) / (num_c + 1)
                        
                        # Histogram binning over t_S breaks
                        breaks = t_S[low_curr : low_next + 1]
                        # np.histogram bins: [breaks[0], breaks[1]), ..., [breaks[-2], breaks[-1]]
                        counts, _ = np.histogram(cen_t, bins=breaks)
                        cen[low_curr : low_next] = counts

                    # Calculate events and at risk on this interval
                    n_hat[low_curr] = n_risk[i]
                    last = last_i[i]

                    for k in range(low_curr, up_curr + 1):
                        if i == 0 and k == low_curr:
                            d[k] = 0
                            KM_hat[k] = 1.0
                        else:
                            denom = max(1e-8, KM_hat[last])
                            d[k] = int(round(n_hat[k] * (1.0 - (S[k] / denom))))
                            d[k] = max(0, min(int(n_hat[k]), d[k]))

                        if n_hat[k] > 0:
                            KM_hat[k] = KM_hat[last] * (1.0 - (float(d[k]) / float(n_hat[k])))
                        else:
                            KM_hat[k] = 0.0

                        n_hat[k + 1] = max(0.0, n_hat[k] - d[k] - cen[k])
                        if d[k] != 0:
                            last = k

                    # Check convergence
                    diff = n_hat[low_next] - n_risk[i + 1]
                    if abs(diff) < 0.5:
                        break
                    
                    if diff < 0 and n_censor[i] <= 0:
                        # Cannot decrease censor count below 0
                        break

                    n_censor[i] = n_censor[i] + diff

                if n_hat[low_next] < n_risk[i + 1]:
                    # Adjust risk if reconstructed at risk is smaller
                    n_risk[i + 1] = int(round(n_hat[low_next]))

                last_i[i + 1] = last

        # Final interval (interval index n_int - 1)
        last_idx = n_int - 1
        low_last = lower[last_idx]
        up_last = upper[last_idx]

        if n_int > 1:
            # Assume same censor rate as average over previous intervals
            prior_censor_sum = np.sum(n_censor[:last_idx])
            prior_time_span = max(1e-6, t_S[upper[last_idx - 1]] - t_S[lower[0]])
            curr_time_span = max(0.0, t_S[up_last] - t_S[low_last])
            extrap_cen = round(prior_censor_sum * (curr_time_span / prior_time_span))
            n_censor[last_idx] = min(max(0, extrap_cen), n_risk[last_idx])
        else:
            n_censor[last_idx] = 0

        if n_censor[last_idx] <= 0:
            cen[low_last : up_last] = 0
            n_censor[last_idx] = 0
        else:
            num_c = int(round(n_censor[last_idx]))
            j_vals = np.arange(1, num_c + 1)
            cen_t = t_S[low_last] + j_vals * (t_S[up_last] - t_S[low_last]) / (num_c + 1)
            breaks = t_S[low_last : up_last + 1]
            counts, _ = np.histogram(cen_t, bins=breaks)
            cen[low_last : up_last] = counts

        # Compute events on final interval
        n_hat[low_last] = n_risk[last_idx]
        last = last_i[last_idx]
        for k in range(low_last, up_last + 1):
            if KM_hat[last] > 1e-8:
                d[k] = int(round(n_hat[k] * (1.0 - (S[k] / KM_hat[last]))))
            else:
                d[k] = 0
            d[k] = max(0, min(int(n_hat[k]), d[k]))

            if n_hat[k] > 0:
                KM_hat[k] = KM_hat[last] * (1.0 - (float(d[k]) / float(n_hat[k])))
            else:
                KM_hat[k] = 0.0

            if k < up_last:
                n_hat[k + 1] = n_hat[k] - d[k] - cen[k]
                if n_hat[k + 1] < 0:
                    cen[k] = max(0, int(n_hat[k] - d[k]))
                    n_hat[k + 1] = 0
            if d[k] != 0:
                last = k

        # Optional: Adjust for reported total events
        if self.tot_events is not None and self.tot_events > 0:
            target_tot = int(self.tot_events)
            sum_dL = np.sum(d[:upper[last_idx - 1] + 1]) if n_int > 1 else 0
            if n_int > 1 and sum_dL >= target_tot:
                # Already exceeded target: zero out events in final interval
                d[low_last : up_last + 1] = 0
                cen[low_last : up_last] = 0
                n_hat[low_last + 1 : up_last + 2] = n_risk[last_idx]
            else:
                sum_d = int(np.sum(d[: up_last + 1]))
                tot_iter = 0
                while tot_iter < 300 and (sum_d > target_tot or (sum_d < target_tot and n_censor[last_idx] > 0)):
                    tot_iter += 1
                    n_censor[last_idx] += (sum_d - target_tot)
                    if n_censor[last_idx] <= 0:
                        cen[low_last : up_last] = 0
                        n_censor[last_idx] = 0
                    else:
                        num_c = int(round(n_censor[last_idx]))
                        j_vals = np.arange(1, num_c + 1)
                        cen_t = t_S[low_last] + j_vals * (t_S[up_last] - t_S[low_last]) / (num_c + 1)
                        breaks = t_S[low_last : up_last + 1]
                        counts, _ = np.histogram(cen_t, bins=breaks)
                        cen[low_last : up_last] = counts

                    n_hat[low_last] = n_risk[last_idx]
                    last = last_i[last_idx]
                    for k in range(low_last, up_last + 1):
                        denom = max(1e-8, KM_hat[last])
                        d[k] = int(round(n_hat[k] * (1.0 - (S[k] / denom))))
                        d[k] = max(0, min(int(n_hat[k]), d[k]))
                        if n_hat[k] > 0:
                            KM_hat[k] = KM_hat[last] * (1.0 - (float(d[k]) / float(n_hat[k])))
                        else:
                            KM_hat[k] = 0.0

                        if k < up_last:
                            n_hat[k + 1] = n_hat[k] - d[k] - cen[k]
                            if n_hat[k + 1] < 0:
                                cen[k] = max(0, int(n_hat[k] - d[k]))
                                n_hat[k + 1] = 0
                        if d[k] != 0:
                            last = k

                    sum_d = int(np.sum(d[: up_last + 1]))

            # Direct reconciliation if total event constraint is specified
            current_tot = int(np.sum(d[:n_t]))
            diff = target_tot - current_tot
            if diff != 0:
                for k in range(up_last, low_last - 1, -1):
                    if diff > 0:
                        avail = int(n_hat[k]) - int(d[k]) - int(cen[k])
                        add_events = min(diff, max(0, avail))
                        d[k] += add_events
                        diff -= add_events
                    elif diff < 0:
                        reduce_events = min(-diff, int(d[k]))
                        d[k] -= reduce_events
                        diff += reduce_events
                    if diff == 0:
                        break

        # Construct patient-level pseudo-IPD
        ipd_rows: List[PseudoIPDRow] = []
        patient_counter = 1
        now_str = datetime.now(timezone.utc).isoformat()
        constraint_str = str(self.tot_events) if self.tot_events is not None else "none"

        # 1. Event rows
        for j in range(n_t):
            num_events = int(d[j])
            t_event = float(t_S[j])
            for _ in range(num_events):
                ipd_rows.append(PseudoIPDRow(
                    patient_id=f"arm{self.arm_id}_pat_{patient_counter:04d}",
                    arm_id=self.arm_id,
                    arm_name=self.arm_name,
                    time=t_event,
                    event=1,
                    reconstruction_method="Guyot et al. 2012",
                    total_events_constraint=constraint_str,
                    created_at=now_str
                ))
                patient_counter += 1

        # 2. Censored rows within intervals
        for j in range(n_t - 1):
            num_censored = int(cen[j])
            # Assign censor time at interval midpoint between t_S[j] and t_S[j+1]
            t_cen = float(round((t_S[j] + t_S[j + 1]) / 2.0, 3))
            for _ in range(num_censored):
                ipd_rows.append(PseudoIPDRow(
                    patient_id=f"arm{self.arm_id}_pat_{patient_counter:04d}",
                    arm_id=self.arm_id,
                    arm_name=self.arm_name,
                    time=t_cen,
                    event=0,
                    reconstruction_method="Guyot et al. 2012",
                    total_events_constraint=constraint_str,
                    created_at=now_str
                ))
                patient_counter += 1

        # 3. Final follow-up censoring: any remaining patients still at risk at the end of the curve
        total_initial_patients = int(n_risk[0])
        current_assigned = len(ipd_rows)
        remaining_at_end = total_initial_patients - current_assigned

        if remaining_at_end > 0:
            t_final = float(t_S[-1])
            for _ in range(remaining_at_end):
                ipd_rows.append(PseudoIPDRow(
                    patient_id=f"arm{self.arm_id}_pat_{patient_counter:04d}",
                    arm_id=self.arm_id,
                    arm_name=self.arm_name,
                    time=t_final,
                    event=0,
                    reconstruction_method="Guyot et al. 2012",
                    total_events_constraint=constraint_str,
                    created_at=now_str
                ))
                patient_counter += 1

        # Build summary DataFrame
        summary_df = pd.DataFrame({
            "time": t_S,
            "S_digitized": S,
            "KM_hat": KM_hat[:n_t],
            "n_hat": n_hat[:n_t],
            "events_d": d[:n_t],
            "censored_cen": cen[:n_t]
        })

        metadata = {
            "initial_sample_size": total_initial_patients,
            "reconstructed_sample_size": len(ipd_rows),
            "total_events_reconstructed": int(np.sum(d[:n_t])),
            "total_censored_reconstructed": int(len(ipd_rows) - np.sum(d[:n_t])),
            "target_total_events": self.tot_events,
            "warnings": self.warnings
        }

        return ipd_rows, summary_df, metadata
