"""
================================================================================
  AEROTWIN B787 — XAI ENGINE  (GradientExplainer — IEEE V5.0)
  Boeing 787-9 Landing Gear | CS-25.473 | ATA 32

  Produces:
    shap_heatmap.png        — per-channel SHAP heatmap (6 × 100 timesteps)
    shap_summary_bar.png    — global feature importance bar chart
    shap_waveform_overlay.png — raw signal + SHAP magnitude overlay (6 panels)

  Why GradientExplainer (not KernelExplainer):
    • Your model is fully differentiable → GradientExplainer uses backprop,
      not perturbation sampling, so it is 100× faster and more precise.
    • KernelExplainer treats the model as a black box and flattens the input
      to (600,) — that is why all the reshape errors kept appearing.
    • GradientExplainer operates natively in the (N, 6, 100) input space.
================================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
import shap

# ─── PATHS — edit these to match your machine ────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'best_model.pt')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'signal_scalers.pkl')
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'sample_dataset.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

N_BACKGROUND = 50    # number of reference samples for GradientExplainer
N_EXPLAIN    = 20    # number of test samples to explain
N_SIGNALS    = 6
N_TIMESTEPS  = 100

CLASS_NAMES = [
    'Normal',       'N2 Leak',      'Worn Seal',    'Early Degrad',
    'Thermal Degrad','Brake Fade',   'Tire Burst',   'Corrosion',
    'Hard Landing', 'Combined',     'Impending AOG','Bogie Pitch'
]

CHANNEL_NAMES = [
    'Strut Force (kN)',
    'Strut Deflection (mm)',
    'Axle Accel (g)',
    'Tire Pressure (bar)',
    'Strut Velocity (m/s)',
    'Brake Torque (Nm)'
]

# ─── EXACT ARCHITECTURE FROM YOUR TRAINING SCRIPT ────────────────────────────
# (Copied verbatim so the state_dict keys match perfectly.)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights


class AEROTWIN_CNN_LSTM(nn.Module):
    def __init__(self, n_channels=6, n_timesteps=100, n_classes=12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=2,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.attention = TemporalAttention(hidden_dim=256)
        self.trunk = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head_class    = nn.Linear(128, n_classes)
        self.head_severity = nn.Linear(128, 4)
        self.head_rul = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        context, _ = self.attention(lstm_out)
        shared = self.trunk(context)
        return self.head_class(shared), self.head_severity(shared), self.head_rul(shared)


# ─── WRAPPER: GradientExplainer needs a single-output model ──────────────────
# It can only explain one output tensor at a time.
# We wrap the model to return only head_class (the classification logits).

class ClassificationWrapper(nn.Module):
    """Returns only the classification logits — required by GradientExplainer."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        cls_out, _, _ = self.base(x)
        return cls_out   # shape: (B, 12)


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
def load_model(model_path, device):
    print(f"\n[1/5] Loading model checkpoint: {model_path}")
    if not os.path.exists(model_path):
        sys.exit(f"ERROR: Model not found at {model_path}\n"
                 f"  Check that training completed and best_model.pt was saved.")

    checkpoint = torch.load(model_path, map_location=device)

    # Your training script saves a dict with 'model_state' key
    if isinstance(checkpoint, dict):
        state_dict = checkpoint['model_state']
        print(f"  Checkpoint epoch : {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val accuracy     : {checkpoint.get('val_acc', 0):.1%}")
        print(f"  RUL MAE          : {checkpoint.get('rul_mae', 0):.1f}h")
    else:
        # Fallback: raw state_dict (from final_model.pt)
        state_dict = checkpoint
        print("  Loaded raw state_dict (no metadata)")

    model = AEROTWIN_CNN_LSTM(
        n_channels=N_SIGNALS,
        n_timesteps=N_TIMESTEPS,
        n_classes=12
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded successfully.")
    return model


# ─── LOAD & PREPROCESS DATA ──────────────────────────────────────────────────
def load_data(data_path, scaler_path):
    print(f"\n[2/5] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Shape: {df.shape}")

    # Raw signals: columns 32–631 → reshape to (N, 6, 100)
    X_raw = df.iloc[:, 32:632].values.reshape(-1, N_SIGNALS, N_TIMESTEPS).astype(np.float32)
    y     = df['Class'].values

    # Apply per-channel StandardScaler fitted during training
    X_scaled = np.zeros_like(X_raw)
    if os.path.exists(scaler_path):
        print(f"  Applying saved scalers: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        for ch, sc in enumerate(scalers):
            X_scaled[:, ch, :] = sc.transform(
                X_raw[:, ch, :].reshape(-1, 1)
            ).reshape(-1, N_TIMESTEPS)
    else:
        print("  WARNING: scaler not found — using unscaled signals.")
        X_scaled = X_raw

    print(f"  Loaded {len(X_scaled)} samples across {len(np.unique(y))} classes.")
    return X_scaled, y, X_raw   # return raw too for plotting raw waveforms


# ─── MAIN SHAP ANALYSIS ──────────────────────────────────────────────────────
def run_shap(model, X_scaled, y_true, X_raw, device):
    wrapped_model = ClassificationWrapper(model).to(device)
    wrapped_model.eval()

    # ── Select background and test samples ───────────────────────────────────
    # Background: random 50 samples (representative of the training distribution)
    rng = np.random.RandomState(42)
    bg_idx   = rng.choice(len(X_scaled), N_BACKGROUND, replace=False)
    # Test:     one sample per class (up to N_EXPLAIN total)
    test_idx = []
    for cls_id in range(12):
        candidates = np.where(y_true == cls_id)[0]
        if len(candidates) > 0:
            test_idx.append(rng.choice(candidates))
    test_idx = np.array(test_idx[:N_EXPLAIN])

    background = torch.tensor(X_scaled[bg_idx], dtype=torch.float32).to(device)
    test_data  = torch.tensor(X_scaled[test_idx], dtype=torch.float32).to(device)

    print(f"\n[3/5] Running GradientExplainer...")
    print(f"  Background samples : {N_BACKGROUND}")
    print(f"  Test samples       : {len(test_idx)}")

    # GradientExplainer — operates on raw (N, 6, 100) tensors natively
    explainer   = shap.GradientExplainer(wrapped_model, background)
    # shap_values shape: (N_test, N_classes, N_channels, N_timesteps) 
    # i.e., (≤12, 12, 6, 100)
    shap_values = explainer.shap_values(test_data)
    shap_values = np.array(shap_values)  # ensure numpy

    print(f"  SHAP array shape   : {shap_values.shape}")
    # Actual shape: (N_test, N_channels, N_timesteps, N_classes)
    # i.e., (11, 6, 100, 12) — index as shap_values[sample, channel, time, class]

    return shap_values, test_idx, test_data.cpu().numpy(), X_raw[test_idx]


# ─── FIGURE 1: Per-channel heatmap for the predicted class ───────────────────
def plot_shap_heatmap(shap_values, test_data_scaled, test_idx, y_true,
                      wrapped_model, device, out_dir):
    print("\n[4/5] Generating plots...")

    # Get predicted class for sample 0 (Hard Landing if present)
    with torch.no_grad():
        logits = wrapped_model(
            torch.tensor(test_data_scaled[:1], dtype=torch.float32).to(device)
        )
    pred_class = int(logits.argmax(1).cpu().item())
    true_class = int(y_true[test_idx[0]])
    print(f"  Sample 0 — True: {CLASS_NAMES[true_class]}  "
          f"| Predicted: {CLASS_NAMES[pred_class]}")

    # shap_values shape: (N_test, 6, 100, N_classes)
    sv = shap_values[0, :, :, pred_class]   # (6, 100)
    raw_signal = test_data_scaled[0]  # (6, 100) — scaled

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [1, 1.6]})
    fig.suptitle(
        f'AeroTwin IEEE V5.0 — Kernel-SHAP Attribution  '
        f'(ATA-32 Signals)\n'
        f'True: {CLASS_NAMES[true_class]}  |  '
        f'Predicted: {CLASS_NAMES[pred_class]}  |  '
        f'GradientExplainer on CNN-LSTM',
        fontsize=12, fontweight='bold'
    )

    # ── Top panel: raw signal waveform (channel 0 — Strut Force) ─────────────
    t = np.linspace(0, 4, N_TIMESTEPS)   # 4-second landing window
    ax0 = axes[0]
    ax0.plot(t, raw_signal[0], color='#1e40af', lw=1.8, label=CHANNEL_NAMES[0])
    ax0.set_xlabel('Time after touchdown (s)', fontsize=10)
    ax0.set_ylabel('Normalised amplitude', fontsize=10)
    ax0.set_title('Reference Signal — Strut Force Channel', fontsize=10)
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)

    # Annotate Hertzian contact phase (0.6–1.4 s)
    ax0.axvspan(0.6, 1.4, alpha=0.15, color='orange',
                label='Hertzian contact phase')
    ax0.legend(fontsize=9)

    # ── Bottom panel: SHAP heatmap (6 channels × 100 timesteps) ─────────────
    ax1 = axes[1]
    vmax = np.abs(sv).max() or 1e-8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax1.imshow(sv, aspect='auto', cmap='RdBu_r',
                      norm=norm,
                      extent=[0, 4, -0.5, N_SIGNALS - 0.5])
    ax1.set_yticks(range(N_SIGNALS))
    ax1.set_yticklabels(CHANNEL_NAMES, fontsize=9)
    ax1.set_xlabel('Time after touchdown (s)', fontsize=10)
    ax1.set_title(
        f'SHAP Attribution Heatmap — Class: {CLASS_NAMES[pred_class]}\n'
        f'Red = pushes toward this fault class  |  '
        f'Blue = pushes away from it',
        fontsize=10
    )
    plt.colorbar(im, ax=ax1, label='SHAP value', pad=0.01)

    # Annotate Hertzian contact phase on heatmap
    ax1.axvline(0.6, color='orange', lw=1.5, linestyle='--', alpha=0.8)
    ax1.axvline(1.4, color='orange', lw=1.5, linestyle='--', alpha=0.8)
    ax1.text(1.0, N_SIGNALS - 0.2, 'Hertzian\ncontact',
             ha='center', va='top', fontsize=8,
             color='darkorange', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'shap_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")


# ─── FIGURE 2: Global SHAP summary bar (mean |SHAP| per channel) ─────────────
def plot_summary_bar(shap_values, out_dir):
    # Average |SHAP| over all test samples, all classes, all timesteps
    # shap_values: (12_cls, N_test, 6, 100)
    # shap_values: (N_test, 6, 100, N_classes) → mean over samples, timesteps, classes → (6,)
    mean_abs = np.mean(np.abs(shap_values), axis=(0, 2, 3))   # → (6,)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, N_SIGNALS))
    bars = ax.barh(CHANNEL_NAMES, mean_abs, color=colors[::-1], edgecolor='k',
                   linewidth=0.7)
    ax.bar_label(bars, fmt='{:.4f}', padding=3, fontsize=9)
    ax.set_xlabel('Mean |SHAP value| (averaged over samples, classes, timesteps)',
                  fontsize=10)
    ax.set_title('AeroTwin — Global SHAP Feature Importance\n'
                 'ATA-32 Signal Channels | GradientExplainer',
                 fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'shap_summary_bar.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")


# ─── FIGURE 3: Waveform + SHAP overlay (6 individual channel panels) ─────────
def plot_waveform_overlay(shap_values, test_data_scaled, test_idx,
                          y_true, pred_class, out_dir):
    sv  = shap_values[0, :, :, pred_class]   # (6, 100)
    sig = test_data_scaled[0]          # (6, 100)
    t   = np.linspace(0, 4, N_TIMESTEPS)

    fig, axes = plt.subplots(N_SIGNALS, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f'AeroTwin — Per-Channel SHAP Waveform Overlay\n'
        f'Fault: {CLASS_NAMES[pred_class]} | '
        f'Colour = SHAP magnitude (red=positive, blue=negative)',
        fontsize=11, fontweight='bold'
    )

    for ch, ax in enumerate(axes):
        s_ch  = sig[ch]
        sv_ch = sv[ch]

        # Background waveform
        ax.plot(t, s_ch, color='#94a3b8', lw=1.2, zorder=1)

        # Overlay coloured SHAP segments
        vmax = np.abs(sv_ch).max() or 1e-8
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        for i in range(len(t) - 1):
            color = plt.cm.RdBu_r(norm(sv_ch[i]))
            ax.fill_between(t[i:i+2], s_ch[i:i+2],
                            color=color, alpha=0.55, zorder=2)

        ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=8)
        ax.grid(True, alpha=0.2)

        # Mark Hertzian phase
        ax.axvspan(0.6, 1.4, alpha=0.07, color='orange', zorder=0)

        # SHAP peak annotation
        peak_t = t[np.argmax(np.abs(sv_ch))]
        peak_v = s_ch[np.argmax(np.abs(sv_ch))]
        ax.annotate('SHAP peak', xy=(peak_t, peak_v),
                    xytext=(peak_t + 0.3, peak_v * 1.05),
                    arrowprops=dict(arrowstyle='->', color='#ef4444'),
                    fontsize=7, color='#ef4444')

    axes[-1].set_xlabel('Time after touchdown (s)', fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'shap_waveform_overlay.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")


# ─── COMPUTE PEARSON r (for paper claim) ─────────────────────────────────────
def compute_attention_shap_correlation(model, shap_values, test_data_scaled,
                                       pred_class, device):
    """
    Computes Pearson r between temporal attention weights and SHAP magnitudes
    — cited in the paper as r = 0.87.
    """
    wrapped = ClassificationWrapper(model).to(device)
    x_torch = torch.tensor(test_data_scaled[:1], dtype=torch.float32).to(device)

    # Extract attention weights from the LSTM output
    with torch.no_grad():
        feat = model.cnn(x_torch)                   # (1, 128, 25)
        feat = feat.permute(0, 2, 1)                # (1, 25, 128)
        lstm_out, _ = model.lstm(feat)              # (1, 25, 256)
        _, attn_w = model.attention(lstm_out)       # (1, 25, 1)
        attn_w = attn_w.squeeze().cpu().numpy()     # (25,) — one per LSTM step

    # SHAP magnitude aggregated to the same 25-timestep LSTM resolution
    # SHAP is (6, 100) → average over channels, downsample 100→25 via reshape
    sv_ch = np.abs(shap_values[0, :, :, pred_class])      # (6, 100)
    sv_mean = sv_ch.mean(axis=0)                    # (100,)
    # Downsample 100 → 25 (matching MaxPool1d x2 inside CNN)
    sv_25 = sv_mean.reshape(25, 4).mean(axis=1)    # (25,)

    from scipy.stats import pearsonr
    r, p = pearsonr(attn_w, sv_25)
    return r, p


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  AEROTWIN B787 — XAI ENGINE  |  GradientExplainer")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device : {device}")
    if device.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # 1. Load model
    model = load_model(MODEL_PATH, device)

    # 2. Load data
    X_scaled, y_true, X_raw = load_data(DATA_PATH, SCALER_PATH)

    # 3. Run SHAP
    shap_values, test_idx, test_data_scaled, test_raw = \
        run_shap(model, X_scaled, y_true, X_raw, device)

    # Determine predicted class for sample 0 (used in plots 1 & 3)
    wrapped = ClassificationWrapper(model).to(device)
    with torch.no_grad():
        logits = wrapped(
            torch.tensor(test_data_scaled[:1], dtype=torch.float32).to(device)
        )
    pred_class = int(logits.argmax(1).cpu().item())

    # 4. Generate all three figures
    plot_shap_heatmap(shap_values, test_data_scaled, test_idx,
                      y_true, wrapped, device, OUT_DIR)
    plot_summary_bar(shap_values, OUT_DIR)
    plot_waveform_overlay(shap_values, test_data_scaled, test_idx,
                          y_true, pred_class, OUT_DIR)

    # 5. Pearson correlation for paper
    print("\n[5/5] Computing attention–SHAP Pearson correlation...")
    r, p = compute_attention_shap_correlation(
        model, shap_values, test_data_scaled, pred_class, device
    )
    print(f"  Pearson r  = {r:.3f}")
    print(f"  p-value    = {p:.4f}")
    note = "(strong)" if abs(r) >= 0.7 else "(moderate — re-check training)"
    print(f"  Interpretation: {note}")
    print(f"\n  >>> Use in paper: Pearson r = {r:.2f} between attention weights "
          f"and SHAP magnitudes.")

    print(f"\n{'=' * 70}")
    print(f"  All outputs saved to: {OUT_DIR}")
    print(f"    shap_heatmap.png          — Figure for XAI subsection")
    print(f"    shap_summary_bar.png      — Global feature importance")
    print(f"    shap_waveform_overlay.png — Per-channel waveform overlay")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()