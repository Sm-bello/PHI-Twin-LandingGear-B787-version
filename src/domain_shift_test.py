"""
================================================================================
  PHI-TWIN: OPERATIONAL DOMAIN-SHIFT STRESS TESTS
  Evaluates Performance Invariance Across Sink Rate, Aircraft Mass & Temperature
================================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from model import PHITwin_CNN_BiLSTM

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'best_model.pt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'signal_scalers.pkl')
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'sample_dataset.csv')

CLASS_IDS   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
CLASS_MAP   = {c: i for i, c in enumerate(CLASS_IDS)}
N_SIGNALS   = 6
N_TIMESTEPS = 100
RUL_MAX_HOURS = 2000.0

def run_domain_shifts():
    print("=" * 70)
    print("  PHI-TWIN: OPERATIONAL DOMAIN-SHIFT EVALUATION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    X_raw = df.iloc[:, 32:632].values.reshape(-1, N_SIGNALS, N_TIMESTEPS)
    y_true_cls = df['Class'].map(CLASS_MAP).values
    y_true_rul = df['RUL_Hours'].values.astype(np.float32)

    sink_rates = df['SinkRate_ms'].values.astype(np.float32)
    masses = df['Mass_kg'].values.astype(np.float32)
    temps = df['Temp_C'].values.astype(np.float32)

    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PHITwin_CNN_BiLSTM().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    X_scaled = np.zeros_like(X_raw, dtype=np.float32)
    for ch in range(N_SIGNALS):
        X_scaled[:, ch, :] = scalers[ch].transform(
            X_raw[:, ch, :].reshape(-1, 1)
        ).reshape(-1, N_TIMESTEPS)

    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        lc, ls, lr, _ = model(x_tensor)
        preds = lc.argmax(1).cpu().numpy()
        pred_ruls = lr.squeeze(1).cpu().numpy() * RUL_MAX_HOURS

    # 1. Sink Rate Envelope
    print("\n--- 1. CS-25.473 SINK RATE ENVELOPE ---")
    sr_bands = [
        ('Soft Landing (<1.5 m/s)', sink_rates < 1.5),
        ('Nominal (1.5 - 2.5 m/s)', (sink_rates >= 1.5) & (sink_rates <= 2.5)),
        ('Hard Landing (>2.5 m/s)', sink_rates > 2.5)
    ]
    for label, mask in sr_bands:
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true_cls[mask], preds[mask]) * 100
            f1 = f1_score(y_true_cls[mask], preds[mask], average='macro') * 100
            mae = mean_absolute_error(y_true_rul[mask], pred_ruls[mask])
            print(f"  {label:30s} (N={np.sum(mask):3d}) | Acc: {acc:6.2f}% | F1: {f1:6.2f}% | RUL MAE: {mae:5.1f} h")

    # 2. Touchdown Gross Weight Envelope
    print("\n--- 2. AIRCRAFT LANDING WEIGHT ENVELOPE ---")
    mass_bands = [
        ('Light Weight (<=165 tonnes)', masses <= 165000),
        ('Nominal Weight (165-185 tonnes)', (masses > 165000) & (masses <= 185000)),
        ('Heavy / Max Landing Wt (>185 tonnes)', masses > 185000)
    ]
    for label, mask in mass_bands:
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true_cls[mask], preds[mask]) * 100
            f1 = f1_score(y_true_cls[mask], preds[mask], average='macro') * 100
            mae = mean_absolute_error(y_true_rul[mask], pred_ruls[mask])
            print(f"  {label:30s} (N={np.sum(mask):3d}) | Acc: {acc:6.2f}% | F1: {f1:6.2f}% | RUL MAE: {mae:5.1f} h")

    # 3. Ambient Thermal Envelope
    print("\n--- 3. AMBIENT TEMPERATURE ENVELOPE ---")
    temp_bands = [
        ('Sub-Zero Cold Weather (<0 deg C)', temps < 0),
        ('Standard ISA (0 to 25 deg C)', (temps >= 0) & (temps <= 25)),
        ('Extreme Heat (>25 deg C)', temps > 25)
    ]
    for label, mask in temp_bands:
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true_cls[mask], preds[mask]) * 100
            f1 = f1_score(y_true_cls[mask], preds[mask], average='macro') * 100
            mae = mean_absolute_error(y_true_rul[mask], pred_ruls[mask])
            print(f"  {label:30s} (N={np.sum(mask):3d}) | Acc: {acc:6.2f}% | F1: {f1:6.2f}% | RUL MAE: {mae:5.1f} h")

if __name__ == '__main__':
    run_domain_shifts()
