"""Kaplan-Meier Curve Digitizer and Pseudo-IPD Reconstruction Web Application.

A biostatistical tool for extracting survival curves from published KM plots
and reconstructing pseudo-individual patient data (pseudo-IPD) using the Guyot et al. (2012) algorithm.
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.models import CalibrationConfig, DigitizedCurvePoint, PseudoIPDRow, QCReport
from src.calibration import AxisCalibrator
from src.digitizer import KMDigitizer
from src.ipd_reconstruction import GuyotIPDReconstructor
from src.qc import QualityControlEngine

# Ensure writable matplotlib config directory
os.environ["MPLCONFIGDIR"] = "/tmp"

st.set_page_config(
    page_title="KM Digitizer & Pseudo-IPD Reconstruction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def hex_to_bgr(hex_str: str) -> list:
    """Convert hex color string #rrggbb to BGR list [b, g, r]."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return [255, 0, 0]
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return [b, g, r]


def bgr_to_hex(bgr: list) -> str:
    """Convert BGR list [b, g, r] to hex string #rrggbb."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def init_session_state():
    """Initialize persistent Streamlit session state variables."""
    if "image_bgr" not in st.session_state:
        st.session_state.image_bgr = None
    if "image_name" not in st.session_state:
        st.session_state.image_name = "km_plot"
    if "calib_bounds" not in st.session_state:
        st.session_state.calib_bounds = None
    if "digitized_curves" not in st.session_state:
        st.session_state.digitized_curves = {}  # {arm_id: List[DigitizedCurvePoint]}
    if "arm_configs" not in st.session_state:
        st.session_state.arm_configs = []  # List[dict]
    if "risk_tables" not in st.session_state:
        st.session_state.risk_tables = {}  # {arm_id: List[dict]}
    if "tot_events_inputs" not in st.session_state:
        st.session_state.tot_events_inputs = {}  # {arm_id: Optional[int]}
    if "ipd_results" not in st.session_state:
        st.session_state.ipd_results = {}  # {arm_id: List[PseudoIPDRow]}
    if "qc_report" not in st.session_state:
        st.session_state.qc_report = None
    if "qc_evals" not in st.session_state:
        st.session_state.qc_evals = None


init_session_state()

# -----------------------------------------------------------------------------
# SIDEBAR: Scientific Disclaimers & Workspace Files
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 KM Digitizer & IPD")
    st.caption("Reconstructing pseudo-IPD via Guyot et al. (2012)")

    st.markdown("---")
    st.subheader("⚠️ Biostatistical Disclaimers")
    st.info(
        """
        - **Pseudo-IPD, Not True IPD**: Outputs represent mathematically reconstructed pseudo-IPD, not the original clinical trial participant dataset.
        - **Approximation**: Coordinate digitization precision depends on source image resolution, anti-aliasing, and axis alignment.
        - **Risk Table Dependency**: Reliable reconstruction requires reported Number-at-Risk counts and, when available, total event counts.
        - **Inferred Censoring**: Censoring events are distributed across intervals and may not reflect individual participant event dates.
        """
    )

    st.markdown("---")
    st.subheader("📁 Sample KM Images")
    sample_files = sorted(glob.glob("*.png") + glob.glob("*.jpeg") + glob.glob("*.jpg"))
    selected_sample = st.selectbox(
        "Load sample image:",
        options=["(Choose a sample image...)"] + sample_files,
        index=0
    )
    if selected_sample != "(Choose a sample image...)":
        if st.button(f"Load '{selected_sample}'", width="stretch"):
            img = cv2.imread(selected_sample)
            if img is not None:
                st.session_state.image_bgr = img
                st.session_state.image_name = Path(selected_sample).stem
                st.session_state.calib_bounds = AxisCalibrator.auto_detect_plot_bounds(img)
                st.session_state.digitized_curves = {}
                st.session_state.arm_configs = []
                st.session_state.risk_tables = {}
                st.session_state.ipd_results = {}
                st.session_state.qc_report = None
                st.session_state.qc_evals = None
                st.success(f"Loaded {selected_sample} successfully!")
                st.rerun()


# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.title("Kaplan-Meier Curve Digitizer & Pseudo-IPD Reconstruction")
st.write(
    "Extract digitized curve coordinates from published Kaplan-Meier plots and reconstruct "
    "individual patient-level time-to-event data with the validated **Guyot et al. (2012)** algorithm."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Image & Calibration",
    "2. Curve Digitization & Editing",
    "3. Number-at-Risk Table",
    "4. IPD Reconstruction & QC",
    "5. Exports & Downloads"
])

# -----------------------------------------------------------------------------
# TAB 1: Image Upload & Calibration
# -----------------------------------------------------------------------------
with tab1:
    st.header("Step 1: Upload Image & Calibrate Axes")
    col_up, col_preview = st.columns([1, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Kaplan-Meier Curve Image (.png, .jpg, .jpeg):",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            new_name = Path(uploaded_file.name).stem
            if st.session_state.image_name != new_name or st.session_state.image_bgr is None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    st.session_state.image_bgr = img
                    st.session_state.image_name = new_name
                    st.session_state.calib_bounds = AxisCalibrator.auto_detect_plot_bounds(img)
                    st.session_state.digitized_curves = {}
                    st.session_state.arm_configs = []
                    st.session_state.risk_tables = {}
                    st.session_state.ipd_results = {}
                    st.session_state.qc_report = None
                    st.session_state.qc_evals = None
                    st.rerun()

    img_bgr = st.session_state.image_bgr

    if img_bgr is None:
        st.warning("Please upload a Kaplan-Meier plot or select one from the sidebar sample images to continue.")
    else:
        h, w = img_bgr.shape[:2]
        
        # Verify existing bounds fit within current image dimensions
        bounds = st.session_state.calib_bounds
        if (
            bounds is None
            or bounds.get("x_max", 0) > w
            or bounds.get("y_orig", 0) > h
            or bounds.get("x_orig", 0) > w
            or bounds.get("y_max", 0) > h
        ):
            bounds = AxisCalibrator.auto_detect_plot_bounds(img_bgr)
            st.session_state.calib_bounds = bounds

        st.subheader("Axis Calibration Settings")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            t_min = st.number_input("Time Min (t_min)", value=0.0, step=1.0)
            t_max = st.number_input("Time Max (t_max)", value=60.0, step=6.0)
        with c2:
            time_unit = st.selectbox("Time Unit", ["months", "weeks", "days", "years"], index=0)
            st.caption("Time scale unit for x-axis")
        with c3:
            st.metric("Y-Axis Survival Range", "0.0 to 1.0")
            st.caption("Standard biostatistical survival probability")
        with c4:
            st.write("")
            st.write("")
            if st.button("🎯 Auto-Detect Axes", width="stretch", type="primary"):
                bounds = AxisCalibrator.auto_detect_plot_bounds(img_bgr)
                st.session_state.calib_bounds = bounds
                st.success("Axes detected successfully!")
                st.rerun()

        # Clamp coordinate values to image limits [0, w] and [0, h]
        safe_x_orig = float(max(0.0, min(float(w), round(bounds.get("x_orig", w * 0.12), 1))))
        safe_x_max = float(max(0.0, min(float(w), round(bounds.get("x_max", w * 0.90), 1))))
        safe_y_orig = float(max(0.0, min(float(h), round(bounds.get("y_orig", h * 0.75), 1))))
        safe_y_max = float(max(0.0, min(float(h), round(bounds.get("y_max", h * 0.12), 1))))

        st.markdown("**Pixel Coordinates of Detected Axes**")
        img_k = st.session_state.image_name
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            x_orig = st.number_input("Y-Axis / Origin X (px)", value=safe_x_orig, min_value=0.0, max_value=float(w), step=1.0, key=f"x_orig_{img_k}",
                                     help="Vertical y-axis line position")
            n1, n2 = st.columns(2)
            with n1:
                if st.button("◄ -1px", key=f"nudge_xo_l_{img_k}", width="stretch"):
                    st.session_state.calib_bounds["x_orig"] = max(0.0, x_orig - 1.0)
                    st.rerun()
            with n2:
                if st.button("► +1px", key=f"nudge_xo_r_{img_k}", width="stretch"):
                    st.session_state.calib_bounds["x_orig"] = min(float(w), x_orig + 1.0)
                    st.rerun()
        with p2:
            x_max = st.number_input("X-Axis Max X (px)", value=safe_x_max, min_value=0.0, max_value=float(w), step=1.0, key=f"x_max_{img_k}",
                                    help="Rightmost extent of the horizontal x-axis")
            n3, n4 = st.columns(2)
            with n3:
                if st.button("◄ -1px", key=f"nudge_xm_l_{img_k}", width="stretch"):
                    st.session_state.calib_bounds["x_max"] = max(0.0, x_max - 1.0)
                    st.rerun()
            with n4:
                if st.button("► +1px", key=f"nudge_xm_r_{img_k}", width="stretch"):
                    st.session_state.calib_bounds["x_max"] = min(float(w), x_max + 1.0)
                    st.rerun()
        with p3:
            y_orig = st.number_input("X-Axis / S=0.0 Y (px)", value=safe_y_orig, min_value=0.0, max_value=float(h), step=1.0, key=f"y_orig_{img_k}",
                                     help="Horizontal x-axis line at base (Survival = 0.0)")
            n5, n6 = st.columns(2)
            with n5:
                if st.button("▲ -1px", key=f"nudge_yo_u_{img_k}", width="stretch"):
                    st.session_state.calib_bounds["y_orig"] = max(0.0, y_orig - 1.0)
                    st.rerun()
            with n6:
                if st.button("▼ +1px", key=f"nudge_yo_d_{img_k}", width="stretch"):
                    st.session_state.calib_bounds["y_orig"] = min(float(h), y_orig + 1.0)
                    st.rerun()
        with p4:
            y_max = st.number_input("Y-Max / S=1.0 Y (px)", value=safe_y_max, min_value=0.0, max_value=float(h), step=1.0, key=f"y_max_{img_k}",
                                    help="Top of the vertical y-axis (Survival = 1.0)")
            n7, n8 = st.columns(2)
            with n7:
                if st.button("▲ -1px", key=f"nudge_ym_u_{img_k}", width="stretch"):
                    st.session_state.calib_bounds["y_max"] = max(0.0, y_max - 1.0)
                    st.rerun()
            with n8:
                if st.button("▼ +1px", key=f"nudge_ym_d_{img_k}", width="stretch"):
                    st.session_state.calib_bounds["y_max"] = min(float(h), y_max + 1.0)
                    st.rerun()

        st.session_state.calib_bounds = {
            "x_orig": x_orig,
            "y_orig": y_orig,
            "x_max": x_max,
            "y_max": y_max
        }

        # Update config: y-axis is strictly 0.0 to 1.0
        calib_config = CalibrationConfig(
            x_orig=x_orig,
            y_orig=y_orig,
            x_max=x_max,
            y_max=y_max,
            t_min=t_min,
            t_max=t_max,
            s_min=0.0,
            s_max=1.0,
            time_unit=time_unit,
            survival_scale="0-1"
        )
        calibrator = AxisCalibrator(calib_config)

        # Validation errors
        val_errors = calib_config.validate()
        if val_errors:
            for err in val_errors:
                st.error(err)

        # Draw overlay
        calib_overlay = calibrator.draw_calibration_overlay(img_bgr)
        st.image(
            cv2.cvtColor(calib_overlay, cv2.COLOR_BGR2RGB),
            caption=f"Calibration Overlay: {st.session_state.image_name} ({w}x{h} px)",
            width="stretch"
        )


# -----------------------------------------------------------------------------
# TAB 2: Curve Digitization & Editing
# -----------------------------------------------------------------------------
with tab2:
    st.header("Step 2: Detect & Manually Correct Survival Curves")

    if st.session_state.image_bgr is None:
        st.warning("Please upload or select an image in Step 1 first.")
    else:
        img_bgr = st.session_state.image_bgr
        calibrator = AxisCalibrator(calib_config)
        digitizer = KMDigitizer(img_bgr, calibrator, source_name=st.session_state.image_name)
        is_colored = digitizer.is_colored_image()

        col_detect_ctrl, col_curves_display = st.columns([1, 2])

        with col_detect_ctrl:
            st.subheader("Curve Detection Engine")
            default_mode_idx = 0 if is_colored else 1
            detect_mode = st.radio(
                "Detection Strategy:",
                options=[
                    "Color-Based Segmentation (for colored lines)",
                    "Multi-Track Line Tracer (for monochrome, solid & dashed lines)"
                ],
                index=default_mode_idx,
                help="Choose Color-Based for multi-color plots, or Multi-Track for black/white or solid vs dashed plots."
            )

            max_arms = st.slider("Number of treatment arms:", min_value=1, max_value=4, value=2)

            with st.expander("Advanced Detection Parameters", expanded=False):
                dash_bridge = st.slider("Dash Bridging Gap (px)", min_value=4, max_value=30, value=16,
                                        help="Connects dashed line segments horizontally.")
                darkness_thresh = st.slider("Darkness Threshold (0=black, 255=white)", min_value=80, max_value=230, value=175,
                                            help="Controls sensitivity for detecting dark curve lines.")
                color_tolerance = st.slider("Color Tolerance (Delta-E)", min_value=20, max_value=90, value=55,
                                            help="Color distance threshold for matching line colors in LAB space.")

            if st.button("🔍 Run Curve Detection", width="stretch", type="primary"):
                if "Color-Based" in detect_mode:
                    detected = digitizer.detect_curve_colors(max_colors=max_arms)
                    st.session_state.arm_configs = detected
                    st.session_state.digitized_curves = {}
                    for arm in detected:
                        arm_pts = digitizer.extract_curve_by_color(
                            target_bgr=arm["color_bgr"],
                            arm_id=arm["arm_id"],
                            arm_name=arm["arm_name"],
                            color_tol=color_tolerance,
                            dash_bridge_len=dash_bridge
                        )
                        st.session_state.digitized_curves[arm["arm_id"]] = arm_pts
                    st.success(f"Extracted {len(detected)} color curves!")
                else:
                    # Multi-track line follower
                    tracks = digitizer.extract_multitrack_curves(
                        num_tracks=max_arms,
                        darkness_thresh=darkness_thresh,
                        dash_bridge_len=dash_bridge
                    )
                    st.session_state.digitized_curves = tracks
                    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                    new_configs = []
                    for arm_id in range(1, max_arms + 1):
                        name = f"Upper Arm (Arm {arm_id})" if arm_id == 1 else f"Lower Arm (Arm {arm_id})"
                        hex_c = palette[(arm_id - 1) % len(palette)]
                        new_configs.append({
                            "arm_id": arm_id,
                            "arm_name": name,
                            "hex": hex_c,
                            "color_bgr": hex_to_bgr(hex_c)
                        })
                    st.session_state.arm_configs = new_configs
                    st.success(f"Extracted {len(tracks)} multi-track curves from monochrome plot!")
                st.rerun()

            st.markdown("---")
            st.subheader("Treatment Arms Configuration")
            if not st.session_state.arm_configs:
                st.session_state.arm_configs = [
                    {"arm_id": 1, "arm_name": "Arm 1", "color_bgr": [255, 0, 0], "hex": "#0000ff"},
                    {"arm_id": 2, "arm_name": "Arm 2", "color_bgr": [0, 0, 255], "hex": "#ff0000"}
                ]

            for idx, arm in enumerate(st.session_state.arm_configs):
                with st.expander(f"Arm {arm['arm_id']}: {arm['arm_name']}", expanded=False):
                    new_name = st.text_input(f"Name", value=arm["arm_name"], key=f"name_{arm['arm_id']}")
                    arm["arm_name"] = new_name
                    color_hex = st.color_picker(f"Display Color", value=arm.get("hex", "#0000ff"), key=f"hex_{arm['arm_id']}")
                    arm["hex"] = color_hex
                    arm["color_bgr"] = hex_to_bgr(color_hex)

                    col_ext, col_clean = st.columns(2)
                    with col_ext:
                        if st.button(f"Re-extract Color", key=f"btn_ext_{arm['arm_id']}", width="stretch"):
                            arm_pts = digitizer.extract_curve_by_color(
                                target_bgr=arm["color_bgr"],
                                arm_id=arm["arm_id"],
                                arm_name=arm["arm_name"],
                                color_tol=color_tolerance,
                                dash_bridge_len=dash_bridge
                            )
                            st.session_state.digitized_curves[arm["arm_id"]] = arm_pts
                            st.rerun()
                    with col_clean:
                        if st.button(f"Smooth Curve", key=f"btn_simp_{arm['arm_id']}", width="stretch"):
                            current_pts = st.session_state.digitized_curves.get(arm["arm_id"], [])
                            if current_pts:
                                simplified = KMDigitizer.simplify_curve(current_pts, tolerance=0.015)
                                simplified = KMDigitizer.enforce_monotonicity(simplified)
                                st.session_state.digitized_curves[arm["arm_id"]] = simplified
                                st.rerun()

                    # Shift adjustments
                    col_up, col_dn = st.columns(2)
                    with col_up:
                        if st.button(f"Shift +2% S", key=f"btn_sh_up_{arm['arm_id']}", width="stretch"):
                            current_pts = st.session_state.digitized_curves.get(arm["arm_id"], [])
                            if current_pts:
                                st.session_state.digitized_curves[arm["arm_id"]] = KMDigitizer.shift_curve(current_pts, 0.02)
                                st.rerun()
                    with col_dn:
                        if st.button(f"Shift -2% S", key=f"btn_sh_dn_{arm['arm_id']}", width="stretch"):
                            current_pts = st.session_state.digitized_curves.get(arm["arm_id"], [])
                            if current_pts:
                                st.session_state.digitized_curves[arm["arm_id"]] = KMDigitizer.shift_curve(current_pts, -0.02)
                                st.rerun()

        with col_curves_display:
            preview_mode = st.radio(
                "Curve Preview Display:",
                options=["Overlay directly on Original Image", "Clean Step Function Plot", "Side-by-Side Comparison"],
                horizontal=True
            )

            arm_colors_map = {arm["arm_id"]: arm.get("hex", "#0000ff") for arm in st.session_state.arm_configs}

            if preview_mode == "Overlay directly on Original Image":
                overlay_img = digitizer.draw_curve_overlay_on_image(
                    st.session_state.digitized_curves,
                    arm_colors_map
                )
                st.image(
                    cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                    caption="Digitized Curves Overlaid on Original Plot",
                    width="stretch"
                )
            elif preview_mode == "Clean Step Function Plot":
                fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
                ax.set_facecolor("#fcfcfc")
                has_curves = False
                for arm in st.session_state.arm_configs:
                    pts = st.session_state.digitized_curves.get(arm["arm_id"], [])
                    if pts:
                        has_curves = True
                        t_vals = [p.time for p in pts]
                        s_vals = [p.survival for p in pts]
                        ax.step(t_vals, s_vals, where="post", color=arm.get("hex", "#0000ff"),
                                linewidth=2, label=f"{arm['arm_name']} ({len(pts)} pts)")
                        ax.scatter(t_vals, s_vals, color=arm.get("hex", "#0000ff"), s=12, alpha=0.6)

                ax.set_xlim(calib_config.t_min, calib_config.t_max)
                ax.set_ylim(-0.02, 1.05)
                ax.set_xlabel(f"Time ({calib_config.time_unit})")
                ax.set_ylabel("Survival Probability")
                ax.set_title("Digitized Kaplan-Meier Curves")
                ax.grid(True, linestyle=":", alpha=0.5)
                if has_curves:
                    ax.legend(loc="upper right", frameon=True)
                st.pyplot(fig)
                plt.close(fig)
            else:
                # Side by side
                s1, s2 = st.columns(2)
                with s1:
                    overlay_img = digitizer.draw_curve_overlay_on_image(
                        st.session_state.digitized_curves,
                        arm_colors_map
                    )
                    st.image(
                        cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB),
                        caption="Overlay on Image",
                        width="stretch"
                    )
                with s2:
                    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=120)
                    ax.set_facecolor("#fcfcfc")
                    for arm in st.session_state.arm_configs:
                        pts = st.session_state.digitized_curves.get(arm["arm_id"], [])
                        if pts:
                            t_vals = [p.time for p in pts]
                            s_vals = [p.survival for p in pts]
                            ax.step(t_vals, s_vals, where="post", color=arm.get("hex", "#0000ff"),
                                    linewidth=2, label=f"{arm['arm_name']}")
                    ax.set_xlim(calib_config.t_min, calib_config.t_max)
                    ax.set_ylim(-0.02, 1.05)
                    ax.set_xlabel(f"Time ({calib_config.time_unit})")
                    ax.set_ylabel("Survival")
                    ax.grid(True, linestyle=":", alpha=0.5)
                    ax.legend(loc="upper right", frameon=True)
                    st.pyplot(fig)
                    plt.close(fig)

        # Interactive Table Editor for Manual Corrections
        st.markdown("---")
        st.subheader("Manual Point Editor (Add, Remove, or Adjust Coordinates)")
        if st.session_state.arm_configs:
            selected_edit_arm_idx = st.selectbox(
                "Select Arm to Edit Coordinates:",
                options=range(len(st.session_state.arm_configs)),
                format_func=lambda i: f"Arm {st.session_state.arm_configs[i]['arm_id']}: {st.session_state.arm_configs[i]['arm_name']}"
            )
            curr_arm = st.session_state.arm_configs[selected_edit_arm_idx]
            arm_pts = st.session_state.digitized_curves.get(curr_arm["arm_id"], [])

            if arm_pts:
                pts_df = pd.DataFrame([
                    {
                        "index": idx,
                        "time": p.time,
                        "survival": p.survival,
                        "confidence": p.confidence
                    } for idx, p in enumerate(arm_pts)
                ])

                st.write(f"Editing **{curr_arm['arm_name']}** ({len(pts_df)} points). Edit cells directly, add rows, or delete rows:")
                edited_df = st.data_editor(
                    pts_df,
                    num_rows="dynamic",
                    width="stretch",
                    key=f"editor_arm_{curr_arm['arm_id']}"
                )

                col_apply, col_add_pt = st.columns([1, 1])
                with col_apply:
                    if st.button("Apply Changes & Enforce Monotonicity", width="stretch"):
                        new_points = []
                        for _, row in edited_df.iterrows():
                            px, py = calibrator.data_to_pixel(float(row["time"]), float(row["survival"]))
                            new_points.append(DigitizedCurvePoint(
                                arm_id=curr_arm["arm_id"],
                                arm_name=curr_arm["arm_name"],
                                time=float(row["time"]),
                                survival=float(row["survival"]),
                                survival_scale=calib_config.survival_scale,
                                pixel_x=float(px),
                                pixel_y=float(py),
                                source_image=st.session_state.image_name,
                                extraction_method="manual_edited",
                                confidence=float(row.get("confidence", 1.0))
                            ))
                        new_points = KMDigitizer.enforce_monotonicity(new_points)
                        st.session_state.digitized_curves[curr_arm["arm_id"]] = new_points
                        st.success(f"Saved {len(new_points)} points for {curr_arm['arm_name']}!")
                        st.rerun()

                with col_add_pt:
                    with st.popover("➕ Add Specific (Time, Survival) Point"):
                        new_t = st.number_input("Time", min_value=calib_config.t_min, max_value=calib_config.t_max, value=0.0, step=1.0)
                        new_s = st.number_input("Survival (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
                        if st.button("Insert Point", width="stretch"):
                            px, py = calibrator.data_to_pixel(new_t, new_s)
                            curr_points = list(st.session_state.digitized_curves.get(curr_arm["arm_id"], []))
                            curr_points.append(DigitizedCurvePoint(
                                arm_id=curr_arm["arm_id"],
                                arm_name=curr_arm["arm_name"],
                                time=float(round(new_t, 3)),
                                survival=float(round(new_s, 4)),
                                survival_scale=calib_config.survival_scale,
                                pixel_x=float(round(px, 1)),
                                pixel_y=float(round(py, 1)),
                                source_image=st.session_state.image_name,
                                extraction_method="manual_point_inserted",
                                confidence=1.0
                            ))
                            curr_points = KMDigitizer.enforce_monotonicity(curr_points)
                            st.session_state.digitized_curves[curr_arm["arm_id"]] = curr_points
                            st.success(f"Inserted point at ({new_t}, {new_s})!")
                            st.rerun()
            else:
                st.info("No points extracted yet for this arm. Click 'Run Curve Detection' above.")


# -----------------------------------------------------------------------------
# TAB 3: Number-at-Risk Table
# -----------------------------------------------------------------------------
with tab3:
    st.header("Step 3: Number-at-Risk Table & Total Events")
    st.write(
        "Enter the published number-at-risk counts and optional total reported event counts. "
        "The **Guyot et al. (2012)** algorithm uses this information to resolve the joint distribution "
        "of events and interval censoring."
    )

    if not st.session_state.arm_configs:
        st.warning("Please configure arms in Step 2 first.")
    else:
        for arm in st.session_state.arm_configs:
            arm_id = arm["arm_id"]
            arm_name = arm["arm_name"]

            with st.container():
                st.subheader(f"Risk Table for {arm_name} (Arm {arm_id})")

                # Default risk table if not yet created
                if arm_id not in st.session_state.risk_tables or not st.session_state.risk_tables[arm_id]:
                    # Create default interval suggestions based on t_max
                    span = calib_config.t_max - calib_config.t_min
                    step = max(6.0, span / 5.0)
                    times = np.arange(calib_config.t_min, calib_config.t_max + 0.1, step)
                    default_rows = []
                    initial_n = 100
                    for i, t in enumerate(times):
                        default_rows.append({
                            "time": float(round(t, 1)),
                            "n_risk": int(max(0, initial_n - (i * (initial_n // len(times)))))
                        })
                    st.session_state.risk_tables[arm_id] = default_rows

                r_col1, r_col2 = st.columns([2, 1])

                with r_col1:
                    risk_df = pd.DataFrame(st.session_state.risk_tables[arm_id])
                    edited_risk = st.data_editor(
                        risk_df,
                        num_rows="dynamic",
                        width="stretch",
                        key=f"risk_editor_{arm_id}"
                    )
                    st.session_state.risk_tables[arm_id] = edited_risk.to_dict(orient="records")

                with r_col2:
                    st.markdown("**Optional Parameters**")
                    current_tot = st.session_state.tot_events_inputs.get(arm_id, None)
                    tot_input = st.number_input(
                        f"Reported Total Events (optional)",
                        min_value=0,
                        max_value=10000,
                        value=int(current_tot) if current_tot is not None else 0,
                        key=f"tot_events_{arm_id}",
                        help="If reported in the publication text or table, enter total events for this arm."
                    )
                    if tot_input > 0:
                        st.session_state.tot_events_inputs[arm_id] = tot_input
                    else:
                        st.session_state.tot_events_inputs[arm_id] = None

                st.markdown("---")


# -----------------------------------------------------------------------------
# TAB 4: IPD Reconstruction & QC
# -----------------------------------------------------------------------------
with tab4:
    st.header("Step 4: Pseudo-IPD Reconstruction & Quality Control")

    ready_to_reconstruct = True
    for arm in st.session_state.arm_configs:
        arm_id = arm["arm_id"]
        pts = st.session_state.digitized_curves.get(arm_id, [])
        risk = st.session_state.risk_tables.get(arm_id, [])
        if not pts:
            st.warning(f"Arm '{arm['arm_name']}' has no digitized curve points.")
            ready_to_reconstruct = False
        if not risk:
            st.warning(f"Arm '{arm['arm_name']}' has no risk table data.")
            ready_to_reconstruct = False

    col_btn, col_msg = st.columns([1, 2])
    with col_btn:
        run_recon = st.button("🚀 Reconstruct Pseudo-IPD", type="primary", width="stretch")

    if run_recon:
        if not ready_to_reconstruct:
            st.error("Cannot run reconstruction until all arms have curve points and risk tables.")
        else:
            with st.spinner("Reconstructing pseudo-IPD via Guyot algorithm and calculating QC metrics..."):
                all_ipds = {}
                reconstruct_success = True

                for arm in st.session_state.arm_configs:
                    arm_id = arm["arm_id"]
                    pts = st.session_state.digitized_curves[arm_id]
                    risk = st.session_state.risk_tables[arm_id]
                    tot_e = st.session_state.tot_events_inputs.get(arm_id, None)

                    try:
                        reconstructor = GuyotIPDReconstructor(
                            arm_id=arm_id,
                            arm_name=arm["arm_name"],
                            curve_points=pts,
                            risk_table=risk,
                            tot_events=tot_e
                        )
                        ipd_rows, summary_df, meta = reconstructor.reconstruct()
                        all_ipds[arm_id] = ipd_rows
                    except Exception as ex:
                        st.error(f"Reconstruction failed for arm '{arm['arm_name']}': {ex}")
                        reconstruct_success = False

                if reconstruct_success:
                    st.session_state.ipd_results = all_ipds

                    # Generate QC Report
                    arm_names = {arm["arm_id"]: arm["arm_name"] for arm in st.session_state.arm_configs}
                    report, arm_evals = QualityControlEngine.generate_full_report(
                        arm_curves=st.session_state.digitized_curves,
                        arm_ipds=all_ipds,
                        arm_risks=st.session_state.risk_tables,
                        arm_names=arm_names
                    )
                    st.session_state.qc_report = report
                    st.session_state.qc_evals = arm_evals
                    st.success("Pseudo-IPD reconstruction completed successfully!")

    # Display Results & QC
    if st.session_state.ipd_results and st.session_state.qc_report is not None:
        report: QCReport = st.session_state.qc_report
        arm_evals = st.session_state.qc_evals

        st.subheader("Quality Control Metrics")
        for arm in st.session_state.arm_configs:
            arm_id = arm["arm_id"]
            ev = arm_evals.get(arm_id, {})
            if ev:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Arm", ev["arm_name"])
                m2.metric("Sample Size (N)", ev["sample_size"])
                m3.metric("Events (d)", ev["events"])
                m4.metric("Censored", ev["censored"])
                m5.metric("Max Absolute KM Error", f"{ev['max_absolute_error']:.4f}")

        if report.warnings:
            st.warning("⚠️ QC Warnings:")
            for w in report.warnings:
                st.write(f"- {w}")

        st.subheader("Reconstructed KM vs Digitized Curves Comparison")
        fig_qc = QualityControlEngine.plot_reconstructed_vs_digitized(
            arm_evals=arm_evals,
            arm_curves=st.session_state.digitized_curves
        )
        st.pyplot(fig_qc)
        plt.close(fig_qc)

        st.subheader("Reconstructed Pseudo-IPD Sample Preview")
        combined_ipd_rows = []
        for arm_id, rows in st.session_state.ipd_results.items():
            for r in rows:
                combined_ipd_rows.append(r.to_dict())

        combined_ipd_df = pd.DataFrame(combined_ipd_rows)
        st.dataframe(combined_ipd_df.head(20), width="stretch")
        st.caption(f"Showing first 20 of {len(combined_ipd_df)} reconstructed individual patient records.")


# -----------------------------------------------------------------------------
# TAB 5: Exports & Downloads
# -----------------------------------------------------------------------------
with tab5:
    st.header("Step 5: Export Data Files")
    st.write("Download the digitized survival curve coordinates, reconstructed pseudo-IPD, and QC report.")

    base_name = st.session_state.image_name or "km_plot"

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    # 1. Digitized Curves CSV
    with col_exp1:
        st.subheader("1. Digitized Curves CSV")
        all_pts_rows = []
        for arm_id, pts in st.session_state.digitized_curves.items():
            for p in pts:
                all_pts_rows.append(p.to_dict())

        if all_pts_rows:
            curves_df = pd.DataFrame(all_pts_rows)
            curves_csv = curves_df.to_csv(index=False)
            filename_curves = f"{base_name}_digitized_curves.csv"
            st.download_button(
                label=f"📥 Download {filename_curves}",
                data=curves_csv,
                file_name=filename_curves,
                mime="text/csv",
                width="stretch"
            )
            st.write(f"Points: {len(curves_df)}")
        else:
            st.info("No curves digitized yet.")

    # 2. Reconstructed Pseudo-IPD CSV
    with col_exp2:
        st.subheader("2. Reconstructed Pseudo-IPD CSV")
        all_ipd_rows = []
        for arm_id, rows in st.session_state.ipd_results.items():
            for r in rows:
                all_ipd_rows.append(r.to_dict())

        if all_ipd_rows:
            ipd_df = pd.DataFrame(all_ipd_rows)
            ipd_csv = ipd_df.to_csv(index=False)
            filename_ipd = f"{base_name}_reconstructed_ipd.csv"
            st.download_button(
                label=f"📥 Download {filename_ipd}",
                data=ipd_csv,
                file_name=filename_ipd,
                mime="text/csv",
                width="stretch"
            )
            st.write(f"Patients: {len(ipd_df)}")
        else:
            st.info("No pseudo-IPD reconstructed yet.")

    # 3. QC Report JSON
    with col_exp3:
        st.subheader("3. QC Report JSON")
        if st.session_state.qc_report is not None:
            qc_json_str = json.dumps(st.session_state.qc_report.to_dict(), indent=2)
            filename_qc = f"{base_name}_qc_report.json"
            st.download_button(
                label=f"📥 Download {filename_qc}",
                data=qc_json_str,
                file_name=filename_qc,
                mime="application/json",
                width="stretch"
            )
            st.write("Ready for download.")
        else:
            st.info("No QC report generated yet.")
