"""
================================================================================
  PHI-TWIN: MODEL EVALUATION & BENCHMARKING ENGINE
  Evaluates Pretrained PHI-Twin Weights on Test Waveforms
================================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error
from model import PHITwin_CNN_BiLSTM

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'best_model.pt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'signal_scalers.pkl')
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'sample_dataset.csv')

CLASS_IDS   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
CLASS_MAP   = {c: i for i, c in enumerate(CLASS_IDS)}
CLASS_NAMES = [
    'Normal', 'N2 Leak', 'Worn Seal', 'Early Degrad',
    'Thermal Degrad', 'Brake Fade', 'Tire Burst', 'Corrosion',
    'Hard Landing', 'Combined', 'Impending AOG', 'Bogie Pitch'
]
N_SIGNALS   = 6
N_TIMESTEPS = 100
RUL_MAX_HOURS = 2000.0

def evaluate():
    print("=" * 70)
    print("  PHI-TWIN: B787 PROGNOSTIC MODEL EVALUATION")
    print("=" * 70)

    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    X_raw = df.iloc[:, 32:632].values.reshape(-1, N_SIGNALS, N_TIMESTEPS)
    y_true_cls = df['Class'].map(CLASS_MAP).values
    y_true_rul = df['RUL_Hours'].values.astype(np.float32)

    # 2. Load Scalers
    print(f"Loading fitted zero-leakage scalers from {SCALER_PATH}...")
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)

    X_scaled = np.zeros_like(X_raw, dtype=np.float32)
    for ch in range(N_SIGNALS):
        X_scaled[:, ch, :] = scalers[ch].transform(
            X_raw[:, ch, :].reshape(-1, 1)
        ).reshape(-1, N_TIMESTEPS)

    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading pre-trained model on {device} from {MODEL_PATH}...")
    model = PHITwin_CNN_BiLSTM().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # 4. Inference
    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        lc, ls, lr, attn_w = model(x_tensor)
        preds = lc.argmax(1).cpu().numpy()
        pred_ruls = lr.squeeze(1).cpu().numpy() * RUL_MAX_HOURS

    # 5. Metrics
    acc = accuracy_score(y_true_cls, preds)
    f1_macro = f1_score(y_true_cls, preds, average='macro')
    mae_rul = mean_absolute_error(y_true_rul, pred_ruls)

    print("\n---------------------- OVERALL EVALUATION RESULTS ----------------------")
    print(f"  Classification Accuracy : {acc * 100:.2f}%")
    print(f"  Macro F1-Score          : {f1_macro * 100:.2f}%")
    print(f"  RUL Mean Absolute Error : {mae_rul:.1f} flight hours")
    print("------------------------------------------------------------------------\n")

    # Present classes present in this evaluation subset
    labels_present = sorted(list(set(y_true_cls)))
    names_present = [CLASS_NAMES[i] for i in labels_present]
    report = classification_report(y_true_cls, preds, labels=labels_present, target_names=names_present, digits=3)
    print("Detailed Classification Report:")
    print(report)

if __name__ == '__main__':
    evaluate()
