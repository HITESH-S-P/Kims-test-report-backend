"""
prompts.py
==========
Builds all prompts sent to Mistral 7B.
Two prompts per patient:
  1. Verification prompt  — check Component 1's predictions
  2. Narrative prompt     — generate clinical summary
"""

from features import IDX_TO_CONDITION, IDX_TO_SPECIALIST, ORGAN_FIELDS

# ── Specialist names ───────────────────────────────────────────────
SPECIALIST_NAMES = {
    1: "Cardiologist",
    2: "Nephrologist",
    3: "Gastroenterologist",
    4: "Hepatologist",
    5: "Pulmonologist",
    6: "Hematologist",
    7: "Endocrinologist",
    8: "Neurologist",
    9: "Urologist",
    10: "General Medicine / Infectious Disease Specialist",
}

# ── Normal ranges for flag generation ─────────────────────────────
NORMAL_RANGES = {
    'hemoglobin':      {'male': (13.0, 17.0), 'female': (12.0, 15.0)},
    'wbc':             (4000, 11000),
    'platelets':       (150000, 400000),
    'creatinine':      {'male': (0.7, 1.2),   'female': (0.6, 1.1)},
    'urea':            (17, 43),
    'albumin':         (3.5, 5.2),
    'bilirubin_total': (0.2, 1.2),
    'hba1c':           (4.0, 5.6),
    'sgot':            (0, 40),
    'sgpt':            (0, 40),
    'alp':             (44, 147),
    'sodium':          (136, 146),
    'potassium':       (3.5, 5.1),
    'tsat':            (20, 50),
    'ferritin':        {'male': (22, 322), 'female': (10, 291)},
    'ef_percent':      (55, 80),
    'spo2':            (95, 100),
    'bp_systolic':     (90, 140),
    'bp_diastolic':    (60, 90),
}


def _flag(field, value, gender_str):
    """Return ▲ HIGH, ▼ LOW, or '' for a lab value."""
    if value is None:
        return ''
    r = NORMAL_RANGES.get(field)
    if r is None:
        return ''
    if isinstance(r, dict):
        r = r.get(gender_str, r.get('male'))
    lo, hi = r
    if value < lo:   return '▼ LOW'
    if value > hi:   return '▲ HIGH'
    return ''


def _format_labs(labs, mask, gender_str):
    """Format lab values into a readable list, skipping missing ones."""
    lines = []
    lab_display = [
        ('hemoglobin',      'Haemoglobin',       'g/dL'),
        ('platelets',       'Platelets',          '/cumm'),
        ('wbc',             'WBC',                'cells/cumm'),
        ('esr',             'ESR',                'mm/hr'),
        ('bilirubin_total', 'Total Bilirubin',    'mg/dL'),
        ('albumin',         'Albumin',            'g/dL'),
        ('sgot',            'SGOT/AST',           'U/L'),
        ('sgpt',            'SGPT/ALT',           'U/L'),
        ('alp',             'ALP',                'U/L'),
        ('creatinine',      'Creatinine',         'mg/dL'),
        ('urea',            'Urea/BUN',           'mg/dL'),
        ('sodium',          'Sodium',             'mmol/L'),
        ('potassium',       'Potassium',          'mmol/L'),
        ('hba1c',           'HbA1c',              '%'),
        ('grbs',            'Glucose (fasting)',  'mg/dL'),
        ('ferritin',        'Ferritin',           'ng/mL'),
        ('serum_iron',      'Serum Iron',         'µg/dL'),
        ('tsat',            'TSAT',               '%'),
        ('ef_percent',      'Ejection Fraction',  '%'),
    ]
    for field, label, unit in lab_display:
        if mask.get(f'{field}_missing', 1) == 0 and labs.get(field) is not None:
            val  = labs[field]
            flag = _flag(field, val, gender_str)
            flag_str = f"  {flag}" if flag else ''
            lines.append(f"  {label:<25} {val} {unit}{flag_str}")
    return '\n'.join(lines) if lines else '  No lab values available'


def build_verification_prompt(patient: dict, c1_predictions: dict) -> str:
    """
    Build the verification prompt.
    Asks Mistral to check if Component 1's predictions make clinical sense
    and identify anything missed. Returns JSON.
    """
    labs    = patient.get('lab_features', {})
    mask    = patient.get('lab_missing_mask', {})
    dem     = patient.get('demographics', {})
    vit     = patient.get('vitals', {})
    com     = patient.get('comorbidities', {})
    age     = dem.get('age', 0)
    gender  = 'male' if dem.get('gender', 1) == 1 else 'female'
    text_in = patient.get('text_input', '')

    # Component 1 predictions
    cond_idx = c1_predictions.get('condition_idx', 0)
    risk_idx = c1_predictions.get('risk_idx', 1)
    spec_idx = c1_predictions.get('specialist_idx', 9)
    organ_binary = c1_predictions.get('organ_binary', [0]*6)
    cond_conf    = c1_predictions.get('condition_confidence', 0.0)

    cond_name = IDX_TO_CONDITION.get(cond_idx, 'unknown')
    risk_lvl  = risk_idx + 1
    spec_code = IDX_TO_SPECIALIST.get(spec_idx, 10)
    spec_name = SPECIALIST_NAMES.get(spec_code, 'General Medicine')
    organs    = [ORGAN_FIELDS[i] for i, v in enumerate(organ_binary) if v == 1]

    lab_text = _format_labs(labs, mask, gender)

    # Active comorbidities
    active_com = [k.replace('_', ' ') for k, v in com.items() if v == 1]
    com_text   = ', '.join(active_com) if active_com else 'None reported'

    # Vitals
    spo2 = vit.get('spo2')
    bp_s = vit.get('bp_systolic')
    bp_d = vit.get('bp_diastolic')
    pul  = vit.get('pulse')
    vit_parts = []
    if spo2:  vit_parts.append(f"SpO2 {spo2}%{' ▼ LOW' if spo2 < 95 else ''}")
    if bp_s and bp_d: vit_parts.append(f"BP {bp_s}/{bp_d} mmHg{' ▲ HIGH' if bp_s > 140 else ''}")
    if pul:   vit_parts.append(f"Pulse {pul} bpm")
    vit_text = ', '.join(vit_parts) if vit_parts else 'Not available'

    prompt = f"""You are an experienced clinician reviewing an AI-generated clinical assessment.

PATIENT: {age}-year-old {gender}
PRESENTING COMPLAINT: {text_in[:300] if text_in else 'Not provided'}
COMORBIDITIES: {com_text}
VITALS: {vit_text}

LAB RESULTS:
{lab_text}

AI MODEL PREDICTIONS:
  Primary condition: {cond_name} (confidence: {cond_conf:.0%})
  Risk level: {risk_lvl}/4
  Recommended specialist: {spec_name}
  Organ involvement detected: {', '.join(organs) if organs else 'None'}

TASK: Review the lab results and AI predictions above. Respond ONLY with a JSON object in this exact format:
{{
  "predictions_correct": true or false,
  "corrected_condition": "{cond_name} or corrected condition name if wrong",
  "corrected_risk_level": {risk_lvl} or corrected 1-4 if wrong,
  "corrected_specialist": "{spec_name} or corrected specialist if wrong",
  "additional_organ_involvement": ["organ1", "organ2"] or [],
  "missed_findings": ["finding1", "finding2"] or [],
  "critical_flags": ["list only critically abnormal values e.g. Na<125 or K>6 or Hb<7 or Cr>3, otherwise empty array"],
  "confidence_note": "brief note if predictions are uncertain or ambiguous"
}}

Valid condition names: infection, anemia, gi, diabetes, respiratory, cardiac, renal, hepatic
Valid specialists: Cardiologist, Nephrologist, Gastroenterologist, Hepatologist, Pulmonologist, Hematologist, Endocrinologist, Neurologist, Urologist, General Medicine
Respond with JSON only. No explanation, no markdown."""

    return prompt


def build_narrative_prompt(patient: dict, c1_predictions: dict,
                            verification: dict) -> str:
    """
    Build the narrative generation prompt.
    Uses verified predictions to generate a clinical summary paragraph.
    """
    labs    = patient.get('lab_features', {})
    mask    = patient.get('lab_missing_mask', {})
    dem     = patient.get('demographics', {})
    vit     = patient.get('vitals', {})
    age     = dem.get('age', 0)
    gender  = 'male' if dem.get('gender', 1) == 1 else 'female'
    text_in = patient.get('text_input', '')

    # Use verified predictions if available
    cond_name  = verification.get('corrected_condition',
                  IDX_TO_CONDITION.get(c1_predictions.get('condition_idx', 0), 'unknown'))
    risk_lvl   = verification.get('corrected_risk_level',
                  c1_predictions.get('risk_idx', 1) + 1)
    spec_code  = IDX_TO_SPECIALIST.get(c1_predictions.get('specialist_idx', 9), 10)
    spec_name  = verification.get('corrected_specialist',
              SPECIALIST_NAMES.get(spec_code, 'General Medicine'))

    organ_binary   = c1_predictions.get('organ_binary', [0]*6)
    organs         = [ORGAN_FIELDS[i] for i, v in enumerate(organ_binary) if v == 1]
    extra_organs   = verification.get('additional_organ_involvement', [])
    all_organs     = list(set(organs + extra_organs))
    missed         = verification.get('missed_findings', [])
    critical_flags = verification.get('critical_flags', [])
    conf_note      = verification.get('confidence_note', '')

    lab_text = _format_labs(labs, mask, gender)

    risk_words = {1: 'low risk', 2: 'moderate risk', 3: 'high risk', 4: 'critical'}
    risk_word  = risk_words.get(risk_lvl, 'moderate risk')

    prompt = f"""You are a senior clinician writing a structured clinical summary for a patient report.

PATIENT: {age}-year-old {gender}
PRESENTING COMPLAINT: {text_in[:400] if text_in else 'Not provided'}

LAB RESULTS:
{lab_text}

CLINICAL ASSESSMENT:
  Primary condition: {cond_name}
  Risk level: {risk_lvl}/4 ({risk_word})
  Recommended specialist: {spec_name}
  Organ involvement: {', '.join(all_organs) if all_organs else 'None identified'}
  Additional findings: {', '.join(missed) if missed else 'None'}
  Critical flags: {', '.join(critical_flags) if critical_flags else 'None'}
  {'Note: ' + conf_note if conf_note else ''}

Write a professional clinical summary paragraph of 4-6 sentences that:
1. States the patient's age, gender, and primary clinical concern
2. References the specific abnormal lab values that support the diagnosis
3. Mentions any organ involvement or secondary concerns
4. States the recommended specialist and urgency level
5. Suggests immediate next steps or investigations

End with this disclaimer on a new line:
"DISCLAIMER: This is an AI-generated clinical decision support summary. All findings must be reviewed and confirmed by a qualified clinician before any clinical decision is made."

Write in clear clinical language suitable for a doctor. Do not use bullet points."""

    return prompt


if __name__ == '__main__':
    # Quick test of prompt building
    sample = {
        'demographics': {'age': 24.0, 'gender': 0},
        'lab_features': {
            'hemoglobin': 9.5, 'platelets': 60000, 'sgot': 1097,
            'albumin': 2.9, 'sodium': 129, 'creatinine': 0.6,
        },
        'lab_missing_mask': {f'{k}_missing': 0 for k in
                             ['hemoglobin','platelets','sgot','albumin','sodium','creatinine']},
        'vitals': {'spo2': 96, 'bp_systolic': 110, 'bp_diastolic': 70, 'pulse': 110},
        'comorbidities': {'hepatic_disease': 1},
        'text_input': 'Vomiting 4 days, loss of appetite, chest pain, retrosternal burning.',
    }
    c1_preds = {
        'condition_idx': 7, 'risk_idx': 2, 'specialist_idx': 3,
        'organ_binary': [1, 0, 0, 0, 1, 0],
        'condition_confidence': 0.82
    }
    verif = {
        'corrected_condition': 'hepatic', 'corrected_risk_level': 3,
        'corrected_specialist': 'Hepatologist',
        'additional_organ_involvement': [], 'missed_findings': ['pancytopenia'],
        'critical_flags': ['SGOT >1000'], 'confidence_note': ''
    }

    vp = build_verification_prompt(sample, c1_preds)
    np_ = build_narrative_prompt(sample, c1_preds, verif)
    print("=== VERIFICATION PROMPT (first 500 chars) ===")
    print(vp[:500])
    print("\n=== NARRATIVE PROMPT (first 500 chars) ===")
    print(np_[:500])