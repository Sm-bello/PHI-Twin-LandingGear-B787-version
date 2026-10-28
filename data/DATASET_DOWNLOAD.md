# Boeing 787-8 Landing Gear Prognostics Dataset

This directory contains the experimental datasets used to train, validate, and test the **PHI-Twin** framework.

---

## 1. Quick Testing (`sample_dataset.csv`)

For immediate local experimentation, reproduction, and continuous integration testing, a stratified **540-sample benchmark dataset** (`sample_dataset.csv`) is included directly in this repository:
* **Rows:** 540 samples (exactly 45 samples per class across all 12 operational and fault regimes).
* **Columns:** 632 features (Sample metadata, touchdown scalars, and 100 timesteps across 6 raw kinematic channels).
* **File size:** ~5.9 MB.

To test the pre-trained model immediately on this sample dataset:
```bash
python src/evaluate.py
```

---

## 2. Full 18,000-Sample Physics Dataset

The full dataset comprises **18,000 physically validated landing events** (~201 MB CSV) synthesized using the high-fidelity oleo-pneumatic ODE engine calibrated against Simscape Multibody topologies and FlightGear 6-DOF kinematics, bounded by European Aviation Safety Agency (EASA) CS-25.473 limit loads.

### Download Links:
* **Kaggle Dataset:** [AeroTwin: Boeing 787 Landing Gear Physics Dataset](https://doi.org/10.34740/KAGGLE/DSV/15021052)
* **DOI:** `10.34740/KAGGLE/DSV/15021052`

### Placement:
1. Download `AEROTWIN_B787_V4p5_Dataset.csv` from the Kaggle repository.
2. Place the file inside the `data/` directory:
   ```text
   data/AEROTWIN_B787_V4p5_Dataset.csv
   ```
3. Run full-scale training or evaluation:
   ```bash
   python src/train.py
   ```

---

## 3. Data Structure & Zero-Leakage Protocol

Each sample in the dataset contains 632 columns:
* **Columns 0–31:** Metadata and scalar condition variables (SampleID, Class, RUL_Hours, Mass_kg, SinkRate_ms, Friction_Coef, Temp_C, etc.).
* **Columns 32–131:** `Defl_t1` ... `Defl_t100` (Strut Telescopic Stroke, m)
* **Columns 132–231:** `Velo_t1` ... `Velo_t100` (Strut Telescoping Velocity, m/s)
* **Columns 232–331:** `Force_t1` ... `Force_t100` (Ground Contact Reaction Force, N)
* **Columns 332–431:** `Accel_t1` ... `Accel_t100` (Unsprung Axle Vertical Acceleration, m/s²)
* **Columns 432–531:** `Pressure_t1` ... `Pressure_t100` (Internal Nitrogen Chamber Pressure, Pa)
* **Columns 532–631:** `BogieAngle_t1` ... `BogieAngle_t100` (Semi-Levered Bogie Pitch Deviation, rad)

> **Zero-Leakage Discipline:** In accordance with aeronautical certification standards, PHI-Twin ingests **only the 6 kinematic waveform channels (columns 32–631)**. All operational scalars (mass, sink rate, severity grade) are strictly withheld during model inference.
