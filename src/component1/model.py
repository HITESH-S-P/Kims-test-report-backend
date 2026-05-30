"""
model.py
========
Component 1 — Tabular Neural Network for clinical prediction.
Multi-head architecture: one shared backbone, four output heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Small residual block for tabular data."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class Component1(nn.Module):
    """
    Multi-head tabular classifier for clinical prediction.

    Input:  flat feature vector (~87 features)
    Output:
        condition_logits  : [B, 8]   primary condition
        risk_logits       : [B, 4]   risk level
        specialist_logits : [B, 10]  specialist routing
        organ_logits      : [B, 6]   organ involvement (sigmoid)
    """

    def __init__(self, input_dim, dropout=0.3):
        super().__init__()

        # ── Shared backbone ───────────────────────────────────────
        self.input_bn = nn.BatchNorm1d(input_dim)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res1 = ResidualBlock(256, dropout)

        self.compress = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res2 = ResidualBlock(128, dropout)

        # ── Output heads ──────────────────────────────────────────
        # Head 1: primary condition (8 classes)
        self.head_condition = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 8)
        )

        # Head 2: risk level (4 classes, ordinal)
        self.head_risk = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        # Head 3: specialist routing (10 classes)
        self.head_specialist = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 10)
        )

        # Head 4: organ involvement (6 binary outputs, independent)
        self.head_organ = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.backbone(x)
        x = self.res1(x)
        x = self.compress(x)
        x = self.res2(x)

        return {
            'condition':  self.head_condition(x),   # logits
            'risk':       self.head_risk(x),         # logits
            'specialist': self.head_specialist(x),   # logits
            'organ':      self.head_organ(x),        # logits (apply sigmoid for probs)
        }

    def predict(self, x):
        """Inference — returns class indices and probabilities."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            return {
                'condition_idx':    out['condition'].argmax(dim=1),
                'condition_probs':  F.softmax(out['condition'], dim=1),
                'risk_idx':         out['risk'].argmax(dim=1),
                'risk_probs':       F.softmax(out['risk'], dim=1),
                'specialist_idx':   out['specialist'].argmax(dim=1),
                'specialist_probs': F.softmax(out['specialist'], dim=1),
                'organ_probs':      torch.sigmoid(out['organ']),
                'organ_binary':     (torch.sigmoid(out['organ']) > 0.5).float(),
            }


class Component1Loss(nn.Module):
    """
    Combined loss for all four heads.
    Weights tuned so each head contributes meaningfully.
    """

    def __init__(self, condition_weights=None, label_smoothing=0.0):
        super().__init__()

        # Class-weighted CE for primary condition (handles imbalance)
        self.condition_loss = nn.CrossEntropyLoss(
            weight=condition_weights,
            label_smoothing=label_smoothing,
            reduction='none'   # we apply sample weights manually
        )
        self.risk_loss       = nn.CrossEntropyLoss(reduction='none')
        self.specialist_loss = nn.CrossEntropyLoss(reduction='none')
        self.organ_loss      = nn.BCEWithLogitsLoss(reduction='none')

        # Head loss weights
        self.w_condition  = 1.5
        self.w_risk       = 1.0
        self.w_specialist = 0.8
        self.w_organ      = 1.2

    def forward(self, outputs, labels, sample_weights):
        """
        outputs      : dict from Component1.forward()
        labels       : dict with condition_idx, risk_idx,
                       specialist_idx, organ_vec tensors
        sample_weights: [B] tensor of per-sample weights
        """
        sw = sample_weights  # [B]

        # Condition loss (weighted by class + sample confidence)
        l_cond = self.condition_loss(
            outputs['condition'], labels['condition_idx']
        )
        l_cond = (l_cond * sw).mean()

        # Risk loss
        l_risk = self.risk_loss(
            outputs['risk'], labels['risk_idx']
        )
        l_risk = (l_risk * sw).mean()

        # Specialist loss
        l_spec = self.specialist_loss(
            outputs['specialist'], labels['specialist_idx']
        )
        l_spec = (l_spec * sw).mean()

        # Organ loss (per-organ BCE, averaged)
        l_org = self.organ_loss(
            outputs['organ'], labels['organ_vec']
        )
        l_org = (l_org.mean(dim=1) * sw).mean()

        total = (self.w_condition  * l_cond +
                 self.w_risk       * l_risk +
                 self.w_specialist * l_spec +
                 self.w_organ      * l_org)

        return {
            'total':      total,
            'condition':  l_cond.item(),
            'risk':       l_risk.item(),
            'specialist': l_spec.item(),
            'organ':      l_org.item(),
        }


if __name__ == '__main__':
    from features import get_feature_dim
    dim   = get_feature_dim()
    model = Component1(input_dim=dim)
    print(f"Input dim: {dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x   = torch.randn(4, dim)
    out = model(x)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
