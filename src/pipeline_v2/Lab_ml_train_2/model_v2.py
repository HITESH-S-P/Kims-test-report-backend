"""
model_v2.py
===========
Component 1 v2 — adds a 5th head: binary health classifier.

CHANGE FROM v1: new `head_health` outputs 2 logits (healthy vs not-healthy).
This is a SEPARATE head, not a 9th condition class. Rationale:
  - "Is this patient sick at all?" is a different question from
    "which of 8 conditions does this sick patient have?"
  - Keeping it separate means the 8-condition head is never diluted
    by normal examples, preserving the 0.847 condition F1.
  - At inference: health head fires first (gate), condition head only
    matters if health head says "not healthy".

Everything else is identical to v1 so existing fold weights stay
architecture-compatible for the condition/risk/specialist/organ heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class Component1V2(nn.Module):
    def __init__(self, input_dim, dropout=0.4):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        )
        self.res1     = ResidualBlock(256, dropout)
        self.compress = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.res2 = ResidualBlock(128, dropout)

        # Existing heads (unchanged)
        self.head_condition  = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout*0.5), nn.Linear(64,8))
        self.head_risk       = nn.Sequential(nn.Linear(128,32), nn.ReLU(), nn.Linear(32,4))
        self.head_specialist = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(dropout*0.5), nn.Linear(64,10))
        self.head_organ      = nn.Sequential(nn.Linear(128,32), nn.ReLU(), nn.Linear(32,6))

        # NEW: binary health head (healthy vs not-healthy)
        self.head_health     = nn.Sequential(nn.Linear(128,32), nn.ReLU(), nn.Linear(32,2))

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
            'condition':  self.head_condition(x),
            'risk':       self.head_risk(x),
            'specialist': self.head_specialist(x),
            'organ':      self.head_organ(x),
            'health':     self.head_health(x),
        }

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            health_probs = F.softmax(out['health'], dim=1)
            return {
                'condition_idx':    out['condition'].argmax(dim=1),
                'condition_probs':  F.softmax(out['condition'], dim=1),
                'risk_idx':         out['risk'].argmax(dim=1),
                'risk_probs':       F.softmax(out['risk'], dim=1),
                'specialist_idx':   out['specialist'].argmax(dim=1),
                'specialist_probs': F.softmax(out['specialist'], dim=1),
                'organ_probs':      torch.sigmoid(out['organ']),
                'organ_binary':     (torch.sigmoid(out['organ']) > 0.5).float(),
                'health_idx':       out['health'].argmax(dim=1),   # 0=not healthy, 1=healthy
                'health_probs':     health_probs,
                'healthy_confidence': health_probs[:, 1],          # P(healthy)
            }


class Component1V2Loss(nn.Module):
    """
    Combined loss across all 5 heads.

    IMPORTANT MEDICAL SAFETY DESIGN:
      - condition/risk/specialist/organ losses are MASKED for healthy
        patients (a healthy person has no condition to predict, so we
        don't penalise those heads on normal examples).
      - health-head loss applies to ALL examples.
      - health head is weighted HIGH because a false "healthy" on a sick
        patient is the most dangerous error we can make.
    """
    def __init__(self, condition_weights=None, label_smoothing=0.1,
                 health_weights=None):
        super().__init__()
        self.condition_loss  = nn.CrossEntropyLoss(weight=condition_weights,
                                                   label_smoothing=label_smoothing,
                                                   reduction='none')
        self.risk_loss       = nn.CrossEntropyLoss(reduction='none')
        self.specialist_loss = nn.CrossEntropyLoss(reduction='none')
        self.organ_loss      = nn.BCEWithLogitsLoss(reduction='none')
        # Health head — weighted so missing a sick patient hurts more
        self.health_loss     = nn.CrossEntropyLoss(weight=health_weights,
                                                   reduction='none')
        self.w_condition  = 1.5
        self.w_risk       = 1.0
        self.w_specialist = 0.8
        self.w_organ      = 1.2
        self.w_health     = 2.0    # highest — safety critical

    def forward(self, outputs, labels, sample_weights):
        sw         = sample_weights                  # [B]
        is_healthy = labels['is_healthy'].float()    # [B], 1=healthy
        sick_mask  = (1.0 - is_healthy)              # condition heads only on sick

        # Condition / risk / specialist / organ — masked to sick patients
        l_cond = (self.condition_loss(outputs['condition'], labels['condition_idx'])
                  * sw * sick_mask).sum() / (sick_mask.sum() + 1e-6)
        l_risk = (self.risk_loss(outputs['risk'], labels['risk_idx'])
                  * sw * sick_mask).sum() / (sick_mask.sum() + 1e-6)
        l_spec = (self.specialist_loss(outputs['specialist'], labels['specialist_idx'])
                  * sw * sick_mask).sum() / (sick_mask.sum() + 1e-6)
        l_org  = (self.organ_loss(outputs['organ'], labels['organ_vec']).mean(dim=1)
                  * sw * sick_mask).sum() / (sick_mask.sum() + 1e-6)

        # Health head — ALL patients
        l_health = (self.health_loss(outputs['health'], labels['is_healthy'].long())
                    * sw).mean()

        total = (self.w_condition*l_cond + self.w_risk*l_risk +
                 self.w_specialist*l_spec + self.w_organ*l_org +
                 self.w_health*l_health)

        return {
            'total': total,
            'condition': float(l_cond), 'risk': float(l_risk),
            'specialist': float(l_spec), 'organ': float(l_org),
            'health': float(l_health),
        }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, r'D:\Major_Project\project\kims_v3\src\component1')
    try:
        from features import get_feature_dim
        dim = get_feature_dim()
    except Exception:
        dim = 85
    model = Component1V2(input_dim=dim)
    print(f"Input dim: {dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(4, dim)
    out = model(x)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
