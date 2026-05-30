"""
lab_ranges.py
=============
Hardcoded clinical reference ranges (from PSAP reference table + standard sources).
Single source of truth used by:
  - normal-case extractor (build_semisupervised_data.py)
  - rule-based clinical gate (clinical_gate.py)
  - Mistral verification prompt (to stop hallucinated critical flags)

All ranges are gender-aware where it matters.
Each entry maps a canonical v3 schema field → reference range.

CRITICAL_THRESHOLDS define when a value is not just abnormal but
clinically critical (used for risk flagging). These are deliberately
conservative so we never under-flag a sick patient.
"""

# ── Standard reference ranges (canonical field → range) ──────────
# Format: either (low, high) tuple, or {'male':(l,h), 'female':(l,h)}
REFERENCE_RANGES = {
    # Haematology
    'hemoglobin':      {'male': (14.0, 18.0), 'female': (12.0, 16.0)},
    'platelets':       (150000, 350000),
    'wbc':             (4500, 11000),
    'pcv':             {'male': (42.0, 50.0), 'female': (36.0, 45.0)},
    'mcv':             (80.0, 100.0),
    'rdw':             (11.5, 14.5),
    'esr':             (0, 20),
    'neutrophils_pct': (45, 75),
    'lymphocytes_pct': (20, 45),

    # Liver function
    'bilirubin_total': (0.3, 1.2),
    'albumin':         (3.5, 5.0),
    'sgot':            (10, 40),       # AST
    'sgpt':            (10, 40),       # ALT
    'alp':             (30, 120),
    'ggt':             (2, 30),

    # Renal
    'creatinine':      {'male': (0.7, 1.2), 'female': (0.6, 1.1)},
    'urea':            (8, 23),
    'uric_acid':       (4.0, 8.0),

    # Electrolytes
    'sodium':          (136, 142),
    'potassium':       (3.5, 5.0),
    'chloride':        (96, 106),

    # Metabolic
    'grbs':            (70, 140),
    'hba1c':           (4.0, 5.6),
    'bmi':             (18.5, 24.9),

    # Iron studies
    'ferritin':        {'male': (15, 200), 'female': (15, 200)},
    'serum_iron':      (60, 170),
    'tsat':            (20, 50),

    # Inflammatory
    'crp':             (0.0, 3.1),

    # Cardiac
    'ef_percent':      (55, 75),

    # Vitals
    'spo2':            (95, 100),
    'bp_systolic':     (90, 130),
    'bp_diastolic':    (60, 85),
    'pulse':           (60, 100),
    'respiratory_rate':(12, 20),
}

# ── Critical thresholds — beyond these = genuinely dangerous ─────
# Used for risk-level escalation and critical_flags.
# (field, direction, threshold, human_label)
CRITICAL_THRESHOLDS = [
    ('hemoglobin',      'low',  7.0,   'Severe anaemia (Hb < 7)'),
    ('platelets',       'low',  50000, 'Severe thrombocytopenia (Plt < 50k)'),
    ('platelets',       'high', 1000000,'Extreme thrombocytosis (Plt > 1M)'),
    ('wbc',             'high', 30000, 'Severe leukocytosis (WBC > 30k)'),
    ('wbc',             'low',  2000,  'Severe leukopenia (WBC < 2k)'),
    ('creatinine',      'high', 3.0,   'Severe renal impairment (Cr > 3)'),
    ('urea',            'high', 100,   'Severe uraemia (Urea > 100)'),
    ('sodium',          'low',  125,   'Severe hyponatraemia (Na < 125)'),
    ('sodium',          'high', 155,   'Severe hypernatraemia (Na > 155)'),
    ('potassium',       'high', 6.0,   'Severe hyperkalaemia (K > 6)'),
    ('potassium',       'low',  2.5,   'Severe hypokalaemia (K < 2.5)'),
    ('bilirubin_total', 'high', 10.0,  'Severe hyperbilirubinaemia (Bili > 10)'),
    ('sgot',            'high', 500,   'Severe transaminitis (AST > 500)'),
    ('sgpt',            'high', 500,   'Severe transaminitis (ALT > 500)'),
    ('hba1c',           'high', 12.0,  'Very poor glycaemic control (HbA1c > 12)'),
    ('grbs',            'high', 400,   'Severe hyperglycaemia (Glucose > 400)'),
    ('grbs',            'low',  50,    'Severe hypoglycaemia (Glucose < 50)'),
    ('spo2',            'low',  88,    'Severe hypoxia (SpO2 < 88)'),
    ('albumin',         'low',  2.0,   'Severe hypoalbuminaemia (Alb < 2)'),
    ('ef_percent',      'low',  30,    'Severe LV dysfunction (EF < 30)'),
]


def get_range(field, gender_str='female'):
    """Return (low, high) for a field, gender-aware. None if unknown."""
    r = REFERENCE_RANGES.get(field)
    if r is None:
        return None
    if isinstance(r, dict):
        return r.get(gender_str, r.get('female'))
    return r


def classify_value(field, value, gender_str='female'):
    """
    Classify a value as 'low' / 'normal' / 'high' / 'unknown'.
    """
    if value is None:
        return 'unknown'
    rng = get_range(field, gender_str)
    if rng is None:
        return 'unknown'
    lo, hi = rng
    if value < lo:
        return 'low'
    if value > hi:
        return 'high'
    return 'normal'


def is_critical(field, value):
    """
    Return (True, label) if value crosses a critical threshold, else (False, None).
    Gender-independent — critical thresholds are absolute.
    """
    if value is None:
        return False, None
    for f, direction, thresh, label in CRITICAL_THRESHOLDS:
        if f != field:
            continue
        if direction == 'low' and value < thresh:
            return True, label
        if direction == 'high' and value > thresh:
            return True, label
    return False, None


def deviation_severity(field, value, gender_str='female'):
    """
    How far outside normal is this value? Returns:
      0.0  = normal
      0.5  = mildly abnormal (just outside range)
      1.0  = moderately abnormal
      2.0  = critically abnormal
    Used to compute an overall 'abnormality score' for the rule gate.
    """
    if value is None:
        return 0.0

    crit, _ = is_critical(field, value)
    if crit:
        return 2.0

    rng = get_range(field, gender_str)
    if rng is None:
        return 0.0
    lo, hi = rng
    if lo <= value <= hi:
        return 0.0

    # How far outside, relative to range width
    width = (hi - lo) if (hi - lo) > 0 else 1.0
    if value < lo:
        dist = (lo - value) / width
    else:
        dist = (value - hi) / width

    if dist < 0.25:
        return 0.5     # mild
    return 1.0         # moderate


if __name__ == '__main__':
    # Quick sanity test using Mrs Vishalakshi's values
    tests = [
        ('sodium',     127.8, 'female'),
        ('alp',        53.0,  'female'),
        ('hemoglobin', 11.8,  'female'),
        ('sgot',       15.0,  'female'),
        ('creatinine', 0.64,  'female'),
        ('hba1c',      5.6,   'female'),
    ]
    print(f"{'Field':<14}{'Value':>8}  {'Class':<8}{'Severity':<9}{'Critical'}")
    print('-' * 55)
    for f, v, g in tests:
        cls  = classify_value(f, v, g)
        sev  = deviation_severity(f, v, g)
        crit, label = is_critical(f, v)
        print(f"{f:<14}{v:>8}  {cls:<8}{sev:<9}{'YES: '+label if crit else 'no'}")
