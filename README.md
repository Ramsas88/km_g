# Kaplan-Meier Curve Digitizer & Pseudo-IPD Reconstruction Application

A biostatistical Python web application to digitize published Kaplan-Meier (KM) survival curve images, export extracted step-function coordinates to CSV, and reconstruct pseudo-individual patient data (pseudo-IPD) using the validated **Guyot et al. (2012)** algorithm from `km_algoritham.pdf`.

---

## Features

- **Multi-Format Image Support**: Upload `.png`, `.jpg`, or `.jpeg` Kaplan-Meier curve images or load bundled sample images (`km.jpeg`, `km1.png`, `km3.png`, etc.).
- **Interactive Axis Calibration**:
  - Automatic detection of plot boundaries and axes via OpenCV edge and morphological filtering.
  - Interactive calibration overlay with real-time visual crosshairs and axis markers.
  - Configurable data units (`months`, `weeks`, `days`, `years`) and survival scales (`0-1` or `0-100%`).
- **Color Segmentation & Step-Wise Digitization**:
  - Automatic color clustering (K-Means) to detect distinct treatment arms.
  - Column-wise line tracing and anti-aliasing bridging.
  - Step-function survival reconstruction with non-increasing monotonicity enforcement ($S(t_{k+1}) \le S(t_k)$).
  - Confidence score estimation based on image coverage and continuity.
- **Manual Point & Curve Editor**:
  - Interactive table editor (`st.data_editor`) to add, modify, or delete curve points.
  - Curve smoothing and simplification tools.
  - Arm renaming, color pickers, and inclusion/exclusion toggles.
- **Number-at-Risk Table**:
  - Interactive editor for reported risk intervals $(t_{\text{risk}}, n_{\text{risk}})$.
  - Support for optional reported total event counts per arm.
- **Guyot et al. (2012) Pseudo-IPD Reconstruction**:
  - Faithful Python port of the Guyot et al. (BMC Medical Research Methodology) algorithm.
  - Iterative numerical solver matching interval risk counts.
  - Uniform censoring distribution across intervals and final follow-up extrapolation.
- **Quality Control (QC) & Diagnostics**:
  - Re-fits Kaplan-Meier survival curves on the reconstructed pseudo-IPD using `lifelines`.
  - Side-by-side plot comparisons of Digitized vs Reconstructed KM curves.
  - Computes Mean Absolute Error (MAE) and Maximum Absolute Error (Max AE).
- **Deterministic Data Exports**:
  - `<image>_digitized_curves.csv`
  - `<image>_reconstructed_ipd.csv`
  - `<image>_qc_report.json`

---

## Important Scientific Notes & Limitations

> [!IMPORTANT]
> - **Pseudo-IPD, Not Original IPD**: The output data represents reconstructed pseudo-individual patient data generated from published summary curves and risk tables. It does not reproduce the actual clinical trial participant identifiers or unmeasured covariates.
> - **Digitization Approximation**: Coordinates extracted from raster images depend on image resolution, line width, anti-aliasing, and axis alignment.
> - **Risk Table Dependency**: Reconstruction fidelity strongly relies on having reported Number-at-Risk counts at regular follow-up intervals.
> - **Inferred Censoring**: Censoring events are distributed uniformly across time intervals; exact censoring times of individual patients are unknown.
> - **Review Required**: All reconstructed datasets should be reviewed against original publication reports before secondary meta-analyses or health technology assessments (HTA).

---

## Installation & Setup

### 1. Prerequisites
Python 3.10 to 3.14 with `pip` or `uv`.

### 2. Virtual Environment & Dependencies
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```
Or with the project's virtual environment:
```bash
./.venv/bin/streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## Data Schemas

### 1. Digitized Curves CSV (`<image>_digitized_curves.csv`)
| Column | Type | Description |
|---|---|---|
| `arm_id` | Integer | Arm identifier (1, 2, ...) |
| `arm_name` | String | User-defined arm name |
| `time` | Float | Plotted time coordinate |
| `survival` | Float | Plotted survival probability ($0.0 - 1.0$) |
| `survival_scale` | String | Axis scale (`0-1` or `0-100%`) |
| `pixel_x` | Float | Corresponding X pixel coordinate |
| `pixel_y` | Float | Corresponding Y pixel coordinate |
| `source_image` | String | Filename of the source image |
| `extraction_method` | String | Method (`auto_color_segmentation`, `manual_edited`) |
| `confidence` | Float | Estimated extraction confidence ($0.0 - 1.0$) |

### 2. Reconstructed Pseudo-IPD CSV (`<image>_reconstructed_ipd.csv`)
| Column | Type | Description |
|---|---|---|
| `patient_id` | String | Synthetic patient ID (`arm1_pat_0001`, ...) |
| `arm_id` | Integer | Arm identifier |
| `arm_name` | String | Arm name |
| `time` | Float | Observed survival or censoring time |
| `event` | Integer | Event indicator: `1` = event, `0` = censored |
| `reconstruction_method` | String | Algorithm used (`Guyot et al. 2012`) |
| `total_events_constraint` | String | Constraint value or `none` |
| `created_at` | String | ISO 8601 creation timestamp |

### 3. QC Report JSON (`<image>_qc_report.json`)
Contains summary metrics per arm:
- `extracted_points_per_arm`
- `number_at_risk_used`
- `estimated_events_per_arm`
- `estimated_censored_per_arm`
- `reconstructed_sample_size_per_arm`
- `max_absolute_km_error`
- `mean_absolute_km_error`
- `warnings`
- `errors`

---

## Running Automated Tests

Run the unit test suite:
```bash
./.venv/bin/python -m unittest discover tests
```

Tests verify:
1. `test_calibration.py`: Pixel $\leftrightarrow$ data coordinate transformations, boundary clamping, and axis auto-detection.
2. `test_digitizer.py`: Color segmentation, step-function conversion, monotonicity enforcement, and curve simplification.
3. `test_ipd_reconstruction.py`: Guyot et al. reconstruction algorithm, sample size preservation, event/censor counts, and total event constraint handling.
4. `test_qc.py`: Lifelines KM fitting, Max/Mean Absolute Error computation, and QC schema validation.
