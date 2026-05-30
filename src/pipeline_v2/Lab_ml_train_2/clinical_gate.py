r"""
clinical_gate.py
================
The inference-time safety gate. Runs BEFORE the condition model output
is shown to anyone. Implements the Option A + Option B hybrid:

  LAYER A — Rule gate (deterministic, runs first, no model):
     Compute an abnormality score from raw labs using hardcoded ranges.
     - score == 0           → clearly normal
     - 0 < score < THRESHOLD → mildly abnormal
     - score >= THRESHOLD    → clinically significant, pass to model

  LAYER B — Trained health head (from Component1V2):
     The model's own opinion on whether the patient is healthy.

  DECISION LOGIC (medical-safety biased):
     A patient is only declared "no acute concern" if BOTH the rule gate
     AND the health head agree they're healthy. If EITHER thinks the
     patient is sick, we treat them as sick and run the full condition
     pipeline. (We would rather over-investigate than miss a sick patient.)

  TIERED RESPONSE (your requested format):
     Instead of a binary normal/abnormal, produce:
       - "No acute clinical concern detected." (truly clean)
       - "No acute concern. Mildly abnormal: low sodium (127.8).
          Recommend routine follow-up." (minor deviations)
       - full clinical analysis (anything significant)

This module is called by inference.py / app.py. It does NOT call Mistral
itself — it decides WHETHER the Mistral condition pipeline should run,
and if not, returns a tiered normal response directly.
"""

from lab_ranges import (classify_value, is_critical, deviation_severity,
                        get_range, REFERENCE_RANGES)

# Total abnormality score above which we always run the full model
SIGNIFICANCE_THRESHOLD = 2.0

# Health-head probability above which the model considers patient healthy
HEALTH_CONF_THRESHOLD  = 0.70

CORE_HEALTH_LABS = [
    'hemoglobin', 'platelets', 'wbc', 'creatinine', 'urea',
    'sodium', 'potassium', 'bilirubin_total', 'albumin',
    'sgot', 'sgpt', 'alp', 'hba1c', 'grbs', 'esr',
]

# Friendly display names for the tiered message
DISPLAY_NAMES = {
    'hemoglobin':'haemoglobin','platelets':'platelets','wbc':'WBC',
    'creatinine':'creatinine','urea':'urea','sodium':'sodium',
    'potassium':'potassium','bilirubin_total':'bilirubin','albumin':'albumin',
    'sgot':'SGOT/AST','sgpt':'SGPT/ALT','alp':'ALP','hba1c':'HbA1c',
    'grbs':'glucose','esr':'ESR',
}


def compute_rule_assessment(labs, mask, gender_str='female'):
    """
    LAYER A. Pure rule-based assessment from raw labs.

    Returns dict:
      total_score      : sum of deviation severities
      n_measured       : how many core labs present
      mild_findings    : list of (field, value, 'low'/'high') mild deviations
      moderate_findings: list of moderate deviations
      critical_findings: list of (field, value, label) critical values
      verdict          : 'clean' | 'mild' | 'significant'
    """
    total = 0.0
    measured = 0
    mild, moderate, critical = [], [], []

    for field in CORE_HEALTH_LABS:
        if mask.get(f'{field}_missing', 1) == 1:
            continue
        val = labs.get(field)
        if val is None:
            continue
        measured += 1

        crit, label = is_critical(field, val)
        if crit:
            critical.append((field, val, label))
            total += 2.0
            continue

        cls = classify_value(field, val, gender_str)
        if cls in ('low', 'high'):
            sev = deviation_severity(field, val, gender_str)
            total += sev
            if sev <= 0.5:
                mild.append((field, val, cls))
            else:
                moderate.append((field, val, cls))

    # Verdict
    if critical or total >= SIGNIFICANCE_THRESHOLD or moderate:
        verdict = 'significant'
    elif mild:
        verdict = 'mild'
    else:
        verdict = 'clean'

    return {
        'total_score':       round(total, 2),
        'n_measured':        measured,
        'mild_findings':     mild,
        'moderate_findings': moderate,
        'critical_findings': critical,
        'verdict':           verdict,
    }


def _format_finding(field, value, direction):
    name = DISPLAY_NAMES.get(field, field)
    arrow = 'low' if direction == 'low' else 'high'
    return f"{arrow} {name} ({value})"


def build_tiered_response(rule_assessment):
    """
    Build the human-readable tiered normal response.
    Only called when we've decided the patient does NOT need full analysis.
    """
    ra = rule_assessment
    mild = ra['mild_findings']

    if ra['verdict'] == 'clean':
        return {
            'tier':    'no_concern',
            'headline':'No acute clinical concern detected.',
            'detail':  ('All measured laboratory values fall within their '
                        'normal reference ranges. No further action indicated '
                        'based on these results.'),
            'recommendation': 'Routine health maintenance as appropriate for age.',
        }

    # mild tier — list the minor deviations
    findings_text = ', '.join(_format_finding(f, v, d) for f, v, d in mild)
    return {
        'tier':    'mild_findings',
        'headline':'No acute clinical concern detected.',
        'detail':  f'Mildly abnormal findings noted: {findings_text}.',
        'recommendation': ('Recommend routine follow-up with primary care '
                           'physician. These findings are mild and may be '
                           'incidental, but should be correlated clinically.'),
    }


def run_gate(labs, mask, gender_str='female', health_head_output=None):
    """
    THE MAIN ENTRY POINT.

    Args:
      labs, mask       : patient lab features + missing mask
      gender_str       : 'male'/'female'
      health_head_output: optional dict from Component1V2.predict() with
                          'health_idx' (1=healthy) and 'healthy_confidence'.
                          If None, decision uses rule gate only.

    Returns dict:
      run_full_pipeline : bool — should the Mistral condition pipeline run?
      rule_assessment   : the LAYER A result
      tiered_response    : dict (only if run_full_pipeline is False)
      reason            : human explanation of the decision
    """
    rule = compute_rule_assessment(labs, mask, gender_str)

    # LAYER A verdict
    rule_says_sick = rule['verdict'] == 'significant'

    # LAYER B verdict (if model output available)
    model_says_healthy = None
    if health_head_output is not None:
        h_idx  = int(health_head_output.get('health_idx', 0))
        h_conf = float(health_head_output.get('healthy_confidence', 0.0))
        model_says_healthy = (h_idx == 1 and h_conf >= HEALTH_CONF_THRESHOLD)

    # ── DECISION (safety-biased) ───────────────────────────────────
    # Run full pipeline if EITHER source thinks patient is sick.
    if rule_says_sick:
        return {
            'run_full_pipeline': True,
            'rule_assessment':   rule,
            'tiered_response':   None,
            'reason': (f"Rule gate flagged significant findings "
                       f"(score={rule['total_score']}, "
                       f"{len(rule['critical_findings'])} critical, "
                       f"{len(rule['moderate_findings'])} moderate). "
                       f"Running full clinical analysis."),
        }

    # Rule gate thinks not-significant. Now check model (if present).
    if model_says_healthy is False:
        # Model disagrees — it thinks patient is sick. Trust the model (safety).
        return {
            'run_full_pipeline': True,
            'rule_assessment':   rule,
            'tiered_response':   None,
            'reason': ("Rule gate saw only minor deviations, but the health "
                       "model flagged this patient as potentially unwell. "
                       "Running full analysis out of caution."),
        }

    # Both agree (or model unavailable and rule says clean/mild) → tiered normal
    tiered = build_tiered_response(rule)
    return {
        'run_full_pipeline': False,
        'rule_assessment':   rule,
        'tiered_response':   tiered,
        'reason': (f"No significant abnormalities (rule score="
                   f"{rule['total_score']}"
                   + (f", model P(healthy)={health_head_output.get('healthy_confidence',0):.0%}"
                      if health_head_output else "")
                   + "). Returning tiered normal response."),
    }


if __name__ == '__main__':
    # Test on Mrs Vishalakshi (should be tiered normal, mild sodium)
    print("="*60)
    print("TEST 1: Mrs Vishalakshi (healthy outpatient, low Na)")
    print("="*60)
    labs = {
        'hemoglobin':11.8,'platelets':296000,'wbc':6480,'creatinine':0.64,
        'urea':12.0,'sodium':127.8,'potassium':3.6,'bilirubin_total':0.5,
        'albumin':4.2,'sgot':15.0,'sgpt':11.0,'alp':53.0,'hba1c':5.6,'esr':None,
    }
    mask = {f'{k}_missing': (1 if v is None else 0) for k,v in labs.items()}
    result = run_gate(labs, mask, 'female')
    print(f"Run full pipeline? {result['run_full_pipeline']}")
    print(f"Reason: {result['reason']}")
    if result['tiered_response']:
        tr = result['tiered_response']
        print(f"\n  [{tr['tier']}]")
        print(f"  {tr['headline']}")
        print(f"  {tr['detail']}")
        print(f"  → {tr['recommendation']}")

    print("\n" + "="*60)
    print("TEST 2: BHAVYA (genuinely sick, SGOT 1097)")
    print("="*60)
    labs2 = {
        'hemoglobin':9.5,'platelets':60000,'wbc':8000,'creatinine':0.6,
        'urea':18.0,'sodium':129.0,'potassium':4.6,'bilirubin_total':4.3,
        'albumin':2.9,'sgot':1097.0,'sgpt':375.0,'alp':483.0,'hba1c':5.5,
    }
    mask2 = {f'{k}_missing': 0 for k in labs2}
    result2 = run_gate(labs2, mask2, 'female')
    print(f"Run full pipeline? {result2['run_full_pipeline']}")
    print(f"Reason: {result2['reason']}")

    print("\n" + "="*60)
    print("TEST 3: Perfectly healthy (all normal)")
    print("="*60)
    labs3 = {
        'hemoglobin':14.0,'platelets':250000,'wbc':7000,'creatinine':0.9,
        'urea':20.0,'sodium':140.0,'potassium':4.2,'bilirubin_total':0.8,
        'albumin':4.5,'sgot':25.0,'sgpt':22.0,'alp':80.0,'hba1c':5.2,
    }
    mask3 = {f'{k}_missing': 0 for k in labs3}
    result3 = run_gate(labs3, mask3, 'male')
    print(f"Run full pipeline? {result3['run_full_pipeline']}")
    print(f"Reason: {result3['reason']}")
    if result3['tiered_response']:
        tr = result3['tiered_response']
        print(f"\n  [{tr['tier']}]  {tr['headline']}")
        print(f"  {tr['detail']}")
