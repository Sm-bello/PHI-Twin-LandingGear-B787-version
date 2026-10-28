"""
================================================================================
  PHI-TWIN: ADDITIVE WHITE GAUSSIAN NOISE (AWGN) ROBUSTNESS TEST
  Evaluates Diagnostic & Prognostic Resilience under Sensor Noise (inf to 0 dB)
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

def run_noise_sweep():
    print("=" * 70)
    print("  PHI-TWIN: NOISE ROBUSTNESS STRESS TEST (AWGN)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    X_raw = df.iloc[:, 32:632].values.reshape(-1, N_SIGNALS, N_TIMESTEPS)
    y_true_cls = df['Class'].map(CLASS_MAP).values
    y_true_rul = df['RUL_Hours'].values.astype(np.float32)

    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PHITwin_CNN_BiLSTM().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    snr_levels = [None, 30, 25, 20, 15, 10, 5, 0]
    print(f"\n{'SNR Level':18s} | {'Accuracy':10s} | {'Macro F1':10s} | {'RUL MAE':12s}")
    print("-" * 60)

    for snr in snr_levels:
        if snr is None:
            X_noisy_raw = X_raw.copy()
            lbl = "Clean (inf dB)"
        else:
            lbl = f"{snr:2d} dB"
            X_noisy_raw = np.zeros_like(X_raw)
            for i in range(len(X_raw)):
                for ch in range(N_SIGNALS):
                    sig = X_raw[i, ch]
                    sig_pow = np.mean(sig ** 2)
                    if sig_pow < 1e-12: sig_pow = 1e-12
                    noise_pow = sig_pow / (10 ** (snr / 10.0))
                    noise = np.random.normal(0, np.sqrt(noise_pow), N_TIMESTEPS)
                    X_noisy_raw[i, ch] = sig + noise

        X_scaled = np.zeros_like(X_noisy_raw, dtype=np.float32)
        for ch in range(N_SIGNALS):
            X_scaled[:, ch, :] = scalers[ch].transform(
                X_noisy_raw[:, ch, :].reshape(-1, 1)
            ).reshape(-1, N_TIMESTEPS)

        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            lc, ls, lr, _ = model(x_tensor)
            preds = lc.argmax(1).cpu().numpy()
            pred_ruls = lr.squeeze(1).cpu().numpy() * RUL_MAX_HOURS

        acc = accuracy_score(y_true_cls, preds) * 100
        f1  = f1_score(y_true_cls, preds, average='macro') * 100
        mae = mean_absolute_error(y_true_rul, pred_ruls)

        print(f"  {lbl:16s} | {acc:7.2f}%   | {f1:7.2f}%   | {mae:6.1f} h")
    print("-" * 60)

if __name__ == '__main__':
    run_noise_sweep()
