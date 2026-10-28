"""
================================================================================
  PHI-TWIN: MULTI-TASK CNN-BiLSTM NEURAL NETWORK ARCHITECTURE
  Zero-Leakage Kinematic Deep Learning for Aircraft Landing Gear Prognostics
  
  Architecture:
    - 3-Layer 1D-CNN (Local Morphological Receptive Fields)
    - 2-Layer Bidirectional LSTM (Temporal Pattern Extraction)
    - Temporal Self-Attention Mechanism (Hertzian Contact Localization)
    - Multi-Task Shared Trunk:
        Head 1: 12-Class Fault Classifier (Focal Loss)
        Head 2: 4-Tier Severity Classifier (Cross-Entropy Loss)
        Head 3: Remaining Useful Life Regression (L1 Loss, 0-2000 hours)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss to address severe class imbalance in aerospace fault data.
    Down-weights easy well-classified examples and focuses on hard minority classes.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism allowing the network to dynamically 
    weigh critical impact milliseconds (Hertzian contact phase).
    """
    def __init__(self, hidden_dim=256):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights

class PHITwin_CNN_BiLSTM(nn.Module):
    """
    Zero-Leakage Multi-Task CNN-BiLSTM with Attention.
    Inputs:  (Batch, 6, 100) raw kinematic waveform tensor
    Outputs: 
      - logits_class:    (Batch, 12)
      - logits_severity: (Batch, 4)
      - pred_rul:        (Batch, 1) in normalized [0, 1] scale
      - attn_weights:    (Batch, 25, 1)
    """
    def __init__(self, n_channels=6, n_timesteps=100, n_classes=12):
        super(PHITwin_CNN_BiLSTM, self).__init__()

        # Multi-scale 1D-CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),            # -> (B, 64, 50)
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),            # -> (B, 128, 25)
            nn.Dropout(0.2),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True          # 256-dim output
        )

        # Temporal Self-Attention
        self.attention = TemporalAttention(hidden_dim=256)

        # Shared multi-task trunk
        self.trunk = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Output Heads
        self.head_class    = nn.Linear(128, n_classes)
        self.head_severity = nn.Linear(128, 4)
        self.head_rul      = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (Batch, 6, 100)
        feat = self.cnn(x)                       # (Batch, 128, 25)
        feat = feat.permute(0, 2, 1)             # (Batch, 25, 128)
        lstm_out, _ = self.lstm(feat)            # (Batch, 25, 256)
        context, attn_w = self.attention(lstm_out) # (Batch, 256), (Batch, 25, 1)
        shared = self.trunk(context)             # (Batch, 128)

        logits_class = self.head_class(shared)
        logits_sev   = self.head_severity(shared)
        pred_rul     = self.head_rul(shared)

        return logits_class, logits_sev, pred_rul, attn_w
