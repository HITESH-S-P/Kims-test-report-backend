"""
features.py
===========
Converts a patient JSON into a flat numeric feature vector
for Component 1 training and inference.
"""

import numpy as np
from pathlib import Path
import json

# ── All lab fields in fixed order ─────────────────────────────────
CORE_LABS = [
    'hemoglobin','platelets','wbc','esr','bilirubin_total',
    'albumin','ferritin','serum_iron','creatinine','urea',
    'grbs','bmi','hba1c','tsat','ef_percent'
]
V3_EXTRA_LABS = [
    'sodium','potassium','chloride','sgot','sgpt',
    'alp','ggt','crp','uric_acid','pcv','mcv','rdw',
    'neutrophils_pct','lymphocytes_pct'
]
ALL_LABS = CORE_LABS + V3_EXTRA_LABS

# ── Lab normalisation stats (approximate clinical ranges) ──────────
# Used to scale values to roughly [0,1] range
LAB_SCALE = {
    'hemoglobin':      18.0,
    'platelets':       500000.0,
    'wbc':             20000.0,
    'esr':             120.0,
    'bilirubin_total': 20.0,
    'albumin':         5.5,
    'ferritin':        3000.0,
    'serum_iron':      200.0,
    'creatinine':      15.0,
    'urea':            200.0,
    'grbs':            400.0,
    'bmi':             45.0,
    'hba1c':           15.0,
    'tsat':            60.0,
    'ef_percent':      80.0,
    'sodium':          160.0,
    'potassium':       7.0,
    'chloride':        120.0,
    'sgot':            2000.0,
    'sgpt':            2000.0,
    'alp':             1000.0,
    'ggt':             500.0,
    'crp':             200.0,
    'uric_acid':       12.0,
    'pcv':             55.0,
    'mcv':             120.0,
    'rdw':             25.0,
    'neutrophils_pct': 100.0,
    'lymphocytes_pct': 100.0,
}

# ── Label mappings ─────────────────────────────────────────────────
CONDITION_TO_IDX = {
    'infection': 0, 'anemia': 1, 'gi': 2, 'diabetes': 3,
    'respiratory': 4, 'cardiac': 5, 'renal': 6, 'hepatic': 7
}
IDX_TO_CONDITION = {v: k for k, v in CONDITION_TO_IDX.items()}

SPECIALIST_TO_IDX = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    6: 5, 7: 6, 8: 7, 9: 8, 10: 9
}
IDX_TO_SPECIALIST = {v: k for k, v in SPECIALIST_TO_IDX.items()}

ORGAN_FIELDS = ['hepatic','respiratory','cardiac','renal','gi','urological']


def extract_features(patient):
    """
    Convert one patient JSON dict → flat numpy feature vector.
    Returns: features (np.float32 array of shape [N_FEATURES])
    """
    feats = []

    # ── 1. Demographics ───────────────────────────────────────────
    age    = float(patient.get('demographics', {}).get('age', 0) or 0)
    gender = float(patient.get('demographics', {}).get('gender', 0) or 0)
    feats.append(min(age / 100.0, 1.0))   # normalise to [0,1]
    feats.append(gender)

    # ── 2. Lab values + missing mask ──────────────────────────────
    labs = patient.get('lab_features', {})
    mask = patient.get('lab_missing_mask', {})

    for field in ALL_LABS:
        val     = labs.get(field)
        missing = mask.get(f'{field}_missing', 1)
        scale   = LAB_SCALE.get(field, 100.0)

        if val is not None and missing == 0:
            norm_val = min(float(val) / scale, 3.0)  # cap at 3x scale
        else:
            norm_val = 0.0

        feats.append(norm_val)
        feats.append(float(missing))   # 0=measured, 1=missing

    # ── 3. Vitals ─────────────────────────────────────────────────
    vit = patient.get('vitals', {})
    feats.append(min(float(vit.get('spo2') or 98) / 100.0, 1.0))
    feats.append(min(float(vit.get('bp_systolic') or 120) / 200.0, 1.0))
    feats.append(min(float(vit.get('bp_diastolic') or 80) / 120.0, 1.0))
    feats.append(min(float(vit.get('pulse') or 80) / 150.0, 1.0))
    feats.append(min(float(vit.get('respiratory_rate') or 16) / 40.0, 1.0))
    feats.append(float(vit.get('temperature_afebrile') or 1))

    # ── 4. Comorbidities ──────────────────────────────────────────
    com = patient.get('comorbidities', {})
    for field in ['diabetes','hypertension','cardiac_disease','respiratory_disease',
                  'renal_disease','hepatic_disease','urological_condition',
                  'neurological_condition']:
        feats.append(float(com.get(field, 0) or 0))

    # ── 5. History ────────────────────────────────────────────────
    hst = patient.get('history', {})
    for field in ['known_diabetes','alcohol','smoking','diet_vegetarian','sleep_adequate']:
        feats.append(float(hst.get(field, 0) or 0))

    # ── 6. Cardiac findings ───────────────────────────────────────
    crd = patient.get('cardiac_findings', {})
    for field in ['lv_diastolic_dysfunction','ef_normal','murmur']:
        feats.append(float(crd.get(field, 0) or 0))

    # ── 7. Severity features ──────────────────────────────────────
    sev = patient.get('severity_features', {})
    for field in ['age_risk','multi_organ','hypoalbuminemia']:
        feats.append(float(sev.get(field, 0) or 0))

    return np.array(feats, dtype=np.float32)


def extract_labels(patient):
    """
    Extract all label components from one patient JSON.
    Returns dict with:
        condition_idx  : int (0-7)
        risk_idx       : int (0-3)  [risk_level 1-4 → idx 0-3]
        specialist_idx : int (0-9)
        organ_vec      : np.float32 array of shape [6]
        confidence     : float (sample weight)
    """
    tgt = patient.get('targets', {})

    # Primary condition
    pc  = patient.get('primary_condition', 'infection')
    condition_idx = CONDITION_TO_IDX.get(pc, 0)

    # Risk level (1-4 → 0-3)
    risk_raw  = int(tgt.get('risk_level', 2) or 2)
    risk_idx  = max(0, min(risk_raw - 1, 3))

    # Specialist (code 1-10 → idx 0-9)
    spec_raw  = int(tgt.get('specialist', 10) or 10)
    specialist_idx = SPECIALIST_TO_IDX.get(spec_raw, 9)

    # Organ involvement (6 binary values)
    org = patient.get('organ_involvement', {})
    organ_vec = np.array(
        [float(org.get(f, 0) or 0) for f in ORGAN_FIELDS],
        dtype=np.float32
    )

    # Sample weight from label confidence
    confidence = float(patient.get('label_confidence', 0.80) or 0.80)

    return {
        'condition_idx':  condition_idx,
        'risk_idx':       risk_idx,
        'specialist_idx': specialist_idx,
        'organ_vec':      organ_vec,
        'confidence':     confidence,
    }


def get_feature_dim():
    """Return the total number of input features."""
    # 2 (demographics) + 29*2 (labs+mask) + 6 (vitals) + 8 (comorbidities)
    # + 5 (history) + 3 (cardiac) + 3 (severity)
    return 2 + len(ALL_LABS) * 2 + 6 + 8 + 5 + 3 + 3


if __name__ == '__main__':
    # Quick test
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    if test_file:
        p = json.loads(Path(test_file).read_text(encoding='utf-8'))
        feats  = extract_features(p)
        labels = extract_labels(p)
        print(f"Feature vector shape: {feats.shape}")
        print(f"Feature dim expected: {get_feature_dim()}")
        print(f"Labels: condition={IDX_TO_CONDITION[labels['condition_idx']]}",
              f"risk={labels['risk_idx']+1}",
              f"specialist={IDX_TO_SPECIALIST[labels['specialist_idx']]}",
              f"organs={labels['organ_vec']}")
        print(f"Confidence: {labels['confidence']}")
