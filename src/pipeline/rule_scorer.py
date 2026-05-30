"""
rule_scorer.py
==============
Derives condition_scores, organ_involvement, severity_features,
primary_condition, specialist, and risk_level purely from
extracted lab values + patient demographics.

This is the rule-based scoring logic from v3_parser.py's
infer_primary_condition_and_labels(), refactored as a clean
standalone module for use in the Phase 4 inference pipeline.

No discharge summary required — works from lab values alone.
Called by inference.py after pdf_parser.py extracts lab values.
"""

from typing import Optional

# ── Specialist code map (frozen, matches training data) ────────────
SPECIALIST_MAP = {
    'cardiac':     1,
    'renal':       2,
    'gi':          3,
    'hepatic':     4,
    'respiratory': 5,
    'anemia':      6,
    'diabetes':    7,
    'infection':   10,
}

SPECIALIST_NAMES = {
    1:  'Cardiologist',
    2:  'Nephrologist',
    3:  'Gastroenterologist',
    4:  'Hepatologist',
    5:  'Pulmonologist',
    6:  'Hematologist',
    7:  'Endocrinologist',
    8:  'Neurologist',
    9:  'Urologist',
    10: 'General Medicine / Infectious Disease',
}

# ── Population DM prevalence proxy per condition ───────────────────
DM_PREVALENCE_MAP = {
    'diabetes':    0.25,
    'renal':       0.15,
    'cardiac':     0.20,
    'hepatic':     0.08,
    'infection':   0.05,
    'anemia':      0.10,
    'respiratory': 0.12,
    'gi':          0.07,
}


def score_conditions_from_labs(lab_values: dict) -> dict:
    """
    Pure lab-value-based condition scoring.
    No text/discharge summary needed.
    Returns dict of {condition: score}.
    """
    scores = {
        'infection': 0, 'anemia': 0, 'cardiac': 0,
        'respiratory': 0, 'renal': 0, 'hepatic': 0,
        'gi': 0, 'diabetes': 0,
    }

    hb     = lab_values.get('hemoglobin')
    cr     = lab_values.get('creatinine')
    sgot   = lab_values.get('sgot')
    sgpt   = lab_values.get('sgpt')
    hba1c  = lab_values.get('hba1c')
    plt    = lab_values.get('platelets')
    wbc    = lab_values.get('wbc')
    bili   = lab_values.get('bilirubin_total')
    alb    = lab_values.get('albumin')
    na     = lab_values.get('sodium')
    alp    = lab_values.get('alp')
    ggt    = lab_values.get('ggt')
    crp    = lab_values.get('crp')
    ferrit = lab_values.get('ferritin')
    urea   = lab_values.get('urea')
    grbs   = lab_values.get('grbs')
    rdw    = lab_values.get('rdw')
    mcv    = lab_values.get('mcv')
    neutro = lab_values.get('neutrophils_pct')
    lymph  = lab_values.get('lymphocytes_pct')
    esr    = lab_values.get('esr')
    pcv    = lab_values.get('pcv')

    # ── Anemia signals ─────────────────────────────────────────────
    if hb is not None:
        if hb < 7:    scores['anemia'] += 4
        elif hb < 9:  scores['anemia'] += 3
        elif hb < 11: scores['anemia'] += 2
        elif hb < 12: scores['anemia'] += 1
    if plt is not None and plt < 100000:
        scores['anemia'] += 2
    if mcv is not None and mcv < 75:
        scores['anemia'] += 2   # microcytic → iron deficiency / thalassemia
    if rdw is not None and rdw > 15:
        scores['anemia'] += 1
    if ferrit is not None and ferrit < 15:
        scores['anemia'] += 2   # low ferritin → iron deficiency anemia
    if pcv is not None and pcv < 30:
        scores['anemia'] += 2

    # ── Hepatic signals ────────────────────────────────────────────
    if sgot is not None:
        if sgot > 500:  scores['hepatic'] += 4
        elif sgot > 200: scores['hepatic'] += 3
        elif sgot > 80:  scores['hepatic'] += 2
        elif sgot > 40:  scores['hepatic'] += 1
    if sgpt is not None:
        if sgpt > 200:  scores['hepatic'] += 3
        elif sgpt > 80:  scores['hepatic'] += 2
        elif sgpt > 40:  scores['hepatic'] += 1
    if bili is not None:
        if bili > 5:    scores['hepatic'] += 3
        elif bili > 2:  scores['hepatic'] += 2
        elif bili > 1:  scores['hepatic'] += 1
    if alp is not None and alp > 120:
        scores['hepatic'] += 1
    if ggt is not None and ggt > 50:
        scores['hepatic'] += 1
    if alb is not None and alb < 3.0:
        scores['hepatic'] += 2  # hypoalbuminaemia → chronic liver

    # ── Renal signals ──────────────────────────────────────────────
    if cr is not None:
        if cr > 5.0:    scores['renal'] += 4
        elif cr > 3.0:  scores['renal'] += 3
        elif cr > 2.0:  scores['renal'] += 2
        elif cr > 1.2:  scores['renal'] += 1
    if urea is not None:
        if urea > 100:  scores['renal'] += 3
        elif urea > 60: scores['renal'] += 2
        elif urea > 45: scores['renal'] += 1
    if na is not None and na < 130:
        scores['renal'] += 1    # hyponatraemia also seen in renal

    # ── Diabetes signals ───────────────────────────────────────────
    if hba1c is not None:
        if hba1c > 10:  scores['diabetes'] += 4
        elif hba1c > 8: scores['diabetes'] += 3
        elif hba1c > 6.5: scores['diabetes'] += 2
    if grbs is not None:
        if grbs > 300:  scores['diabetes'] += 3
        elif grbs > 200: scores['diabetes'] += 2
        elif grbs > 126: scores['diabetes'] += 1

    # ── Infection signals ──────────────────────────────────────────
    if wbc is not None:
        if wbc > 15000:  scores['infection'] += 3
        elif wbc > 12000: scores['infection'] += 2
        elif wbc > 11000: scores['infection'] += 1
    if neutro is not None and neutro > 80:
        scores['infection'] += 2
    if crp is not None and crp > 10:
        scores['infection'] += 2
    if esr is not None and esr > 60:
        scores['infection'] += 1
    if ferrit is not None and ferrit > 500:
        scores['infection'] += 1   # high ferritin → systemic inflammation

    # ── Cardiac signals ────────────────────────────────────────────
    # Mostly clinical/echo-based; lab proxies:
    if na is not None and na < 130:
        scores['cardiac'] += 1    # hyponatraemia in heart failure
    if alb is not None and alb < 3.0:
        scores['cardiac'] += 1   # low albumin in cardiac cachexia

    # ── Respiratory signals ────────────────────────────────────────
    # Limited lab signals; respiratory is more clinical
    if wbc is not None and wbc > 12000 and neutro is not None and neutro > 75:
        scores['respiratory'] += 1  # could be pneumonia

    # ── GI signals ─────────────────────────────────────────────────
    if alb is not None and alb < 3.0:
        scores['gi'] += 1   # malabsorption / IBD
    if bili is not None and bili > 1 and sgot is not None and sgot < 80:
        scores['gi'] += 1   # raised bili without major transaminitis → GI

    return scores


def derive_labels(
    lab_values: dict,
    age: int,
    symptoms_text: str = '',
) -> dict:
    """
    Full rule-based label derivation from lab values + age.
    Optionally boosts scores from symptoms_text keywords.

    Returns:
        primary_condition : str
        specialist        : int (code 1-10)
        specialist_name   : str
        risk_level        : int (1-4)
        condition_scores  : dict
        organ_involvement : dict (6 binary fields)
        severity_features : dict (3 binary fields)
        population_dm_prevalence : float
    """
    scores = score_conditions_from_labs(lab_values)

    # ── Optional symptom text boost ────────────────────────────────
    if symptoms_text:
        text = symptoms_text.upper()
        keyword_boosts = {
            'infection':   ['FEVER', 'CHILLS', 'SWEATING', 'DENGUE', 'MALARIA',
                            'TYPHOID', 'SEPSIS', 'UTI', 'PNEUMONIA', 'INFECT',
                            'VIRAL', 'COUGH', 'COLD'],
            'anemia':      ['FATIGUE', 'WEAKNESS', 'PALLOR', 'BREATHLESS', 'DIZZY',
                            'TIRED', 'PALE', 'ANEMIA', 'ANAEMIA'],
            'cardiac':     ['CHEST PAIN', 'PALPITATION', 'BREATHLESS', 'EDEMA',
                            'SWELLING', 'HEART', 'CARDIAC'],
            'respiratory': ['COUGH', 'BREATHLESS', 'WHEEZ', 'SPUTUM', 'HEMOPTYSIS',
                            'SHORT OF BREATH', 'RESPIRATORY'],
            'renal':       ['DECREASED URINE', 'OLIGURIA', 'SWELLING', 'FROTHY URINE',
                            'RENAL', 'KIDNEY'],
            'hepatic':     ['JAUNDICE', 'YELLOW', 'VOMITING', 'ABDOMEN PAIN',
                            'ABDOMINAL PAIN', 'NAUSEA', 'LIVER'],
            'gi':          ['VOMITING', 'DIARRHEA', 'ABDOMEN', 'NAUSEA', 'BLOOD STOOL',
                            'MELENA', 'CONSTIPATION', 'GI', 'STOMACH'],
            'diabetes':    ['POLYURIA', 'POLYDIPSIA', 'POLYPHAGIA', 'DIABETES',
                            'SUGAR', 'INCREASED THIRST', 'INCREASED URINATION'],
        }
        for cond, keywords in keyword_boosts.items():
            for kw in keywords:
                if kw in text:
                    scores[cond] += 1
                    break  # one boost per condition from symptoms

    # ── Primary condition ──────────────────────────────────────────
    primary = max(scores, key=lambda k: scores[k])

    # Tie-break: if top two are within 1 point, prefer lab-abnormality-specific
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    if len(sorted_scores) >= 2 and sorted_scores[0][1] - sorted_scores[1][1] <= 1:
        # Lab-evidence tie-break priority
        priority_order = ['hepatic', 'renal', 'anemia', 'diabetes',
                          'infection', 'cardiac', 'respiratory', 'gi']
        for p in priority_order:
            if scores[p] >= sorted_scores[0][1] - 1:
                primary = p
                break

    specialist = SPECIALIST_MAP.get(primary, 10)

    # ── Risk level ─────────────────────────────────────────────────
    hb    = lab_values.get('hemoglobin')
    cr    = lab_values.get('creatinine')
    sgot  = lab_values.get('sgot')
    plt   = lab_values.get('platelets')
    hba1c = lab_values.get('hba1c')
    na    = lab_values.get('sodium')
    bili  = lab_values.get('bilirubin_total')
    alb   = lab_values.get('albumin')

    critical_signs = 0
    if hb and hb < 7:           critical_signs += 1
    if cr and cr > 3.0:         critical_signs += 1
    if sgot and sgot > 500:     critical_signs += 1
    if plt and plt < 50000:     critical_signs += 1
    if hba1c and hba1c > 10:    critical_signs += 1
    if na and na < 125:         critical_signs += 1
    if bili and bili > 10:      critical_signs += 1
    if alb and alb < 2.0:       critical_signs += 1

    max_score = max(scores.values()) if scores else 0
    if critical_signs >= 3:        risk_level = 4
    elif critical_signs >= 2:      risk_level = 3
    elif critical_signs == 1:      risk_level = 3
    elif max_score >= 6:           risk_level = 3
    elif max_score >= 3:           risk_level = 2
    else:                          risk_level = 1

    # ── Organ involvement ──────────────────────────────────────────
    organ_involvement = {
        'hepatic':     1 if (scores['hepatic'] >= 2 or (sgot and sgot > 80)) else 0,
        'respiratory': 1 if scores['respiratory'] >= 2 else 0,
        'cardiac':     1 if scores['cardiac'] >= 2 else 0,
        'renal':       1 if (scores['renal'] >= 2 or (cr and cr > 1.2)) else 0,
        'gi':          1 if scores['gi'] >= 2 else 0,
        'urological':  0,
    }

    # ── Severity features ──────────────────────────────────────────
    severity_features = {
        'age_risk':        1 if age >= 60 else 0,
        'multi_organ':     1 if sum(organ_involvement.values()) >= 2 else 0,
        'hypoalbuminemia': 1 if (alb is not None and alb < 3.0) else 0,
    }

    return {
        'primary_condition':       primary,
        'specialist':              specialist,
        'specialist_name':         SPECIALIST_NAMES[specialist],
        'risk_level':              risk_level,
        'condition_scores':        scores,
        'organ_involvement':       organ_involvement,
        'severity_features':       severity_features,
        'population_dm_prevalence': DM_PREVALENCE_MAP.get(primary, 0.10),
    }


def build_patient_json(
    lab_values:     dict,
    age:            int,
    gender:         int,          # 1=Male, 0=Female
    symptoms_text:  str  = '',
    comorbidities:  dict = None,
    vitals:         dict = None,
    history:        dict = None,
) -> dict:
    """
    Assembles a complete v3 patient JSON from extracted inputs.
    Suitable for direct use with features.extract_features().

    All fields not provided are zeroed/defaulted safely —
    the missing mask handles absent labs correctly.
    """
    CORE_LABS = [
        'hemoglobin', 'platelets', 'wbc', 'esr', 'bilirubin_total',
        'albumin', 'ferritin', 'serum_iron', 'creatinine', 'urea',
        'grbs', 'bmi', 'hba1c', 'tsat', 'ef_percent'
    ]
    V3_EXTRA_LABS = [
        'sodium', 'potassium', 'chloride', 'sgot', 'sgpt',
        'alp', 'ggt', 'crp', 'uric_acid', 'pcv', 'mcv', 'rdw',
        'neutrophils_pct', 'lymphocytes_pct'
    ]
    ALL_LABS = CORE_LABS + V3_EXTRA_LABS

    lab_features = {}
    lab_missing_mask = {}
    for field in ALL_LABS:
        val = lab_values.get(field)
        lab_features[field] = val
        lab_missing_mask[f'{field}_missing'] = 0 if val is not None else 1

    labels = derive_labels(lab_values, age, symptoms_text)

    default_comorbidities = {
        'diabetes': 0, 'hypertension': 0, 'cardiac_disease': 0,
        'respiratory_disease': 0, 'renal_disease': 0, 'hepatic_disease': 0,
        'urological_condition': 0, 'neurological_condition': 0,
    }
    if comorbidities:
        default_comorbidities.update(comorbidities)

    default_vitals = {
        'spo2': None, 'bp_systolic': None, 'bp_diastolic': None,
        'pulse': None, 'respiratory_rate': None, 'temperature_afebrile': 1,
    }
    if vitals:
        default_vitals.update(vitals)

    default_history = {
        'known_diabetes': 0, 'alcohol': 0, 'smoking': 0,
        'diet_vegetarian': 0, 'sleep_adequate': 1,
    }
    if history:
        default_history.update(history)

    return {
        'text_input':         symptoms_text,
        'demographics':       {'age': float(age), 'gender': gender},
        'lab_features':       lab_features,
        'lab_missing_mask':   lab_missing_mask,
        'comorbidities':      default_comorbidities,
        'vitals':             default_vitals,
        'history':            default_history,
        'cardiac_findings':   {'lv_diastolic_dysfunction': 0, 'ef_normal': 1, 'murmur': 0},
        'condition_scores':   labels['condition_scores'],
        'organ_involvement':  labels['organ_involvement'],
        'severity_features':  labels['severity_features'],
        'primary_condition':  labels['primary_condition'],
        'external_features':  {'population_dm_prevalence': labels['population_dm_prevalence']},
        'label_confidence':   0.70,
        'targets': {
            'risk_level': labels['risk_level'],
            'specialist': labels['specialist'],
        },
    }


if __name__ == '__main__':
    # Quick sanity test with BHAVYA-like lab values
    test_labs = {
        'hemoglobin': 9.5, 'platelets': 60000, 'wbc': 5000,
        'sgot': 1097, 'sgpt': 375, 'alp': 483,
        'bilirubin_total': 4.3, 'albumin': 2.9,
        'sodium': 126, 'potassium': 4.6, 'creatinine': 0.6,
        'ferritin': 2000, 'serum_iron': 48.4,
    }
    result = derive_labels(test_labs, age=24, symptoms_text='vomiting jaundice')
    print('=== BHAVYA test ===')
    print(f"Primary: {result['primary_condition']}")
    print(f"Risk:    {result['risk_level']}")
    print(f"Specialist: {result['specialist_name']}")
    print(f"Organs: {result['organ_involvement']}")
    print(f"Severity: {result['severity_features']}")
    print(f"Scores: {result['condition_scores']}")
