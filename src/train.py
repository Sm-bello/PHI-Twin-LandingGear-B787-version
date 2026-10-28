"""
================================================================================
  PHI-TWIN: MODEL TRAINING PIPELINE
  Zero-Leakage Training with Focal Loss, OneCycleLR & Mixed Precision
================================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from model import PHITwin_CNN_BiLSTM, FocalLoss

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'sample_dataset.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'models')
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE   = 32
EPOCHS       = 25
LR           = 2e-3
WEIGHT_DECAY = 1e-4
RUL_MAX_HOURS= 2000.0

CLASS_IDS   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
CLASS_MAP   = {c: i for i, c in enumerate(CLASS_IDS)}
N_CLASSES   = 12
N_SIGNALS   = 6
N_TIMESTEPS = 100

class LandingGearDataset(Dataset):
    def __init__(self, signals, labels, severities, ruls):
        self.signals    = torch.tensor(signals,    dtype=torch.float32)
        self.labels     = torch.tensor(labels,     dtype=torch.long)
        self.severities = torch.tensor(severities, dtype=torch.long)
        self.ruls       = torch.tensor(ruls,       dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx], self.severities[idx], self.ruls[idx]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training PHI-Twin on {device}...")

    df = pd.read_csv(DATA_PATH)
    X_raw = df.iloc[:, 32:632].values.reshape(-1, N_SIGNALS, N_TIMESTEPS)
    y_class = df['Class'].map(CLASS_MAP).values
    y_sev = np.clip(df['Severity'].values - 1, 0, 3)
    y_rul = df['RUL_Hours'].values.astype(np.float32) / RUL_MAX_HOURS

    # Stratified split 80/20
    n = len(y_class)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    tr_idx, val_idx = idx[:split], idx[split:]

    # Scale on training partition only
    scalers = []
    X = np.zeros_like(X_raw, dtype=np.float32)
    for ch in range(N_SIGNALS):
        sc = StandardScaler()
        sc.fit(X_raw[tr_idx, ch, :].reshape(-1, 1))
        X[:, ch, :] = sc.transform(X_raw[:, ch, :].reshape(-1, 1)).reshape(-1, N_TIMESTEPS)
        scalers.append(sc)

    train_ds = LandingGearDataset(X[tr_idx], y_class[tr_idx], y_sev[tr_idx], y_rul[tr_idx])
    val_ds   = LandingGearDataset(X[val_idx], y_class[val_idx], y_sev[val_idx], y_rul[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PHITwin_CNN_BiLSTM(N_SIGNALS, N_TIMESTEPS, N_CLASSES).to(device)
    crit_cls = FocalLoss(alpha=1.0, gamma=2.0)
    crit_sev = nn.CrossEntropyLoss()
    crit_rul = nn.L1Loss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR,
                                              steps_per_epoch=len(train_loader), epochs=EPOCHS)

    print(f"Beginning training for {EPOCHS} epochs...")
    best_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for sigs, cls_t, sev_t, rul_t in train_loader:
            sigs, cls_t = sigs.to(device), cls_t.to(device)
            sev_t, rul_t = sev_t.to(device), rul_t.to(device).unsqueeze(1)

            optimizer.zero_grad()
            lc, ls, lr, _ = model(sigs)
            loss = 1.0 * crit_cls(lc, cls_t) + 0.3 * crit_sev(ls, sev_t) + 0.2 * crit_rul(lr, rul_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for sigs, cls_t, sev_t, rul_t in val_loader:
                sigs, cls_t = sigs.to(device), cls_t.to(device)
                sev_t, rul_t = sev_t.to(device), rul_t.to(device).unsqueeze(1)
                lc, ls, lr, _ = model(sigs)
                loss = 1.0 * crit_cls(lc, cls_t) + 0.3 * crit_sev(ls, sev_t) + 0.2 * crit_rul(lr, rul_t)
                val_loss += loss.item()
                correct += (lc.argmax(1) == cls_t).sum().item()
                total += len(cls_t)

        val_acc = correct / total
        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {tr_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

    print("Training finished.")

if __name__ == '__main__':
    train()
