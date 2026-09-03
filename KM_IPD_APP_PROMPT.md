# Prompt: Kaplan-Meier Curve Digitizer and IPD Reconstruction App

You are a senior full-stack engineer and biostatistics-aware developer. Build an application in this folder that lets a user upload a Kaplan-Meier survival curve image, digitize all plotted survival curves, export the extracted curve coordinates to CSV, and reconstruct pseudo-individual patient data (pseudo-IPD) using the Kaplan-Meier reconstruction algorithm described in `km_algoritham.pdf`.

Use the sample files in this folder for development and testing:

- `km.jpeg`
- `km1.png`
- `km2.png`
- `km3.png`
- `km4.png`
- `km5.png`
- `km6.png`
- `km7.png`
- `km8.png`
- `km9.png`
- `km_algoritham.pdf`

## Core Goal

The app should accept a JPEG or PNG Kaplan-Meier curve image and produce:

1. A CSV containing the extracted plotted curve data for each treatment arm.
2. A reconstructed pseudo-IPD CSV containing one row per inferred patient with time, event indicator, arm, and reconstruction metadata.

Important: The app must clearly describe the output as reconstructed pseudo-IPD, not original patient-level data.

## Recommended Tech Stack

Use a practical Python-first local app stack. Prefer one of these options:

- Primary recommendation: Streamlit, because it is fast to build, easy to run locally, and well suited for file upload, image display, editable tables, plots, and CSV downloads.
- Alternative recommendation: Shiny for Python, if a more structured reactive application is preferred.

Recommended Python libraries:

- App UI: Streamlit or Shiny for Python.
- Interactive image annotation: Streamlit drawable canvas, Plotly image overlays, or Shiny-compatible interactive plotting tools.
- Image processing: OpenCV, NumPy, scikit-image.
- Plot digitization helpers: OpenCV color segmentation, edge detection, skeletonization, optional OCR via Tesseract or EasyOCR.
- Survival/IPD reconstruction: Python implementation of the Guyot-style Kaplan-Meier reconstruction algorithm, adapted from `km_algoritham.pdf`.
- Data output: pandas CSV export.
- Visualization/QC: matplotlib, Plotly, or lifelines for reconstructed Kaplan-Meier checks.

If choosing between Streamlit and Shiny, implement Streamlit first unless the user specifically requests Shiny for Python.

## User Workflow

Build the app around this workflow:

1. User uploads a Kaplan-Meier plot image in `.jpg`, `.jpeg`, or `.png` format.
2. App displays the image in an interactive calibration canvas.
3. User calibrates the axes:
   - Select x-axis origin and x-axis maximum.
   - Select y-axis origin and y-axis maximum.
   - Enter real axis values, for example time 0 to 60 months and survival 0 to 1 or 0 to 100 percent.
4. App detects plotted curves automatically.
5. User can review, rename, include/exclude, and manually correct each detected curve.
6. User enters or confirms number-at-risk data for each arm:
   - Risk table time points.
   - Number at risk at each time point.
   - Optional total events per arm.
7. App exports digitized curve data to CSV.
8. App reconstructs pseudo-IPD for each arm.
9. App exports pseudo-IPD CSV and a quality-control report.

## Required UI Features

Create a clean application UI with:

- Upload panel for images.
- Image preview and calibration canvas.
- Axis calibration controls.
- Detected curve list with color swatches and editable arm names.
- Manual correction tools:
  - Add point.
  - Remove point.
  - Move point.
  - Smooth or simplify curve.
  - Undo and redo.
- Number-at-risk table editor per arm.
- Optional total events input per arm.
- Export buttons for:
  - Digitized curve CSV.
  - Reconstructed pseudo-IPD CSV.
  - QC report JSON or CSV.
- Clear validation messages when required inputs are missing.

Do not hide the need for number-at-risk data. If risk table data are unavailable, allow curve CSV export, but warn that IPD reconstruction will be less reliable or unavailable depending on selected settings.

## Curve Extraction Requirements

Implement image processing that can handle typical published KM plots:

- Multiple colored treatment arms.
- Step-function survival curves.
- Censor marks if visible.
- Grid lines, axes, labels, legends, and number-at-risk tables below plots.
- Survival axis shown as either 0-1 or 0-100 percent.
- Time axis in months, weeks, days, or years.

Suggested extraction pipeline:

1. Preprocess uploaded image:
   - Normalize image size.
   - Remove or reduce background/grid lines.
   - Detect plot area.
   - Preserve colored curve pixels.
2. Segment curves:
   - Use color clustering or thresholding to separate curve colors.
   - Use connected components to isolate plotted lines.
   - Remove axes, text, and legend artifacts.
3. Convert pixels to data coordinates using calibration points.
4. Reconstruct stepwise KM coordinates:
   - Convert detected pixels into ordered x/y points.
   - Preserve horizontal and vertical KM steps.
   - Simplify duplicate points.
   - Enforce non-increasing survival within each arm.
5. Estimate confidence scores:
   - Coverage of detected curve.
   - Number of discontinuities or gaps.
   - Amount of manual correction required.

CSV output for digitized curves should include:

- `arm_id`
- `arm_name`
- `time`
- `survival`
- `survival_scale`
- `pixel_x`
- `pixel_y`
- `source_image`
- `extraction_method`
- `confidence`

## IPD Reconstruction Requirements

Implement the pseudo-IPD reconstruction based on the algorithm in `km_algoritham.pdf`.

The algorithm should:

1. Accept digitized survival points `t.S` and `S` for each arm.
2. Accept published number-at-risk data:
   - `t.risk`
   - `n.risk`
   - lower and upper curve-point indexes for each risk interval.
3. Estimate the number censored per interval.
4. Estimate events at each digitized survival time.
5. Reconstruct patient-level rows:
   - Event rows with `event = 1`.
   - Censor rows with `event = 0`.
   - Arm identifier.
6. Support optional reported total event count and adjust final interval censoring/events to match it where possible.
7. Recalculate a Kaplan-Meier estimate from reconstructed pseudo-IPD for QC.
8. Compare reconstructed KM against the digitized KM curve.

Pseudo-IPD CSV output should include:

- `patient_id`
- `arm_id`
- `arm_name`
- `time`
- `event`
- `reconstruction_method`
- `total_events_constraint`
- `created_at`

QC output should include:

- Number of extracted points per arm.
- Number at risk used per arm and interval.
- Estimated events per arm.
- Estimated censored observations per arm.
- Reconstructed sample size per arm.
- Maximum absolute KM curve error.
- Mean absolute KM curve error.
- Warnings about missing risk-table data, poor curve detection, invalid monotonic survival, or unresolved axis calibration.

## Validation Rules

Add validation for:

- Unsupported image type.
- Missing axis calibration.
- Survival values outside 0-1 after normalization.
- Non-monotonic KM survival curve.
- Missing or inconsistent number-at-risk values.
- Number at risk increasing unexpectedly across intervals.
- Estimated negative at-risk counts.
- Event or censor counts that are impossible.

Where possible, automatically fix minor issues, but record the correction in the QC report.

## Expected Outputs

The app should create downloadable files:

1. `digitized_curves.csv`
2. `reconstructed_ipd.csv`
3. `qc_report.json`

Use deterministic filenames with the uploaded image base name when practical, for example:

- `km1_digitized_curves.csv`
- `km1_reconstructed_ipd.csv`
- `km1_qc_report.json`

## Testing

Add tests that cover:

- Pixel-to-data coordinate calibration.
- Monotonic survival cleanup.
- Multi-arm curve extraction on sample images.
- Number-at-risk interval mapping.
- Event and censor count reconstruction.
- Pseudo-IPD generation.
- CSV schema correctness.
- QC metric calculation.

Use the sample KM images in this folder as fixtures where possible.

## Acceptance Criteria

The project is complete when:

- A user can upload a KM image and see it in the app.
- The user can calibrate axes interactively.
- At least one sample image from this folder can be digitized into curve CSV.
- The user can enter number-at-risk data.
- The app can reconstruct pseudo-IPD CSV from digitized curve data and risk-table data.
- The app generates a QC report comparing reconstructed KM against digitized KM.
- The implementation includes clear errors and warnings for missing inputs.
- The app has tests for the core algorithmic pieces.
- The README explains installation, running locally, the reconstruction assumptions, and limitations.

## Important Scientific Notes

Include these notes in the README and user-facing help text:

- Image-derived KM extraction is approximate and depends on image quality, axis calibration, and curve clarity.
- IPD reconstructed from KM curves is pseudo-IPD, not the original clinical trial dataset.
- Reliable IPD reconstruction usually requires the published number-at-risk table and, when available, total event counts.
- Censoring times are inferred and may not match true censoring times.
- All outputs should be reviewed before use in statistical analysis.
