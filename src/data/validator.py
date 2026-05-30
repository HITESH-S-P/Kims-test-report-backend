"""
validator.py
============
Validates every JSON in matched_pairs/ and discharge_summaries/
against the v3 schema. Flags issues, auto-fixes what it can,
reports what needs manual attention.
"""

import json, re
from pathlib import Path
from copy import deepcopy

# ── V3 Schema definition ───────────────────────────────────────────
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

REQUIRED_COMORBIDITIES = [
    'diabetes','hypertension','cardiac_disease','respiratory_disease',
    'renal_disease','hepatic_disease','urological_condition','neurological_condition'
]
REQUIRED_CONDITION_SCORES = [
    'infection','anemia','gi','diabetes','respiratory','cardiac','renal','hepatic'
]
REQUIRED_ORGAN_INVOLVEMENT = [
    'hepatic','respiratory','cardiac','renal','gi','urological'
]
VALID_PRIMARY_CONDITIONS = {
    'infection','anemia','gi','diabetes','respiratory','cardiac','renal','hepatic'
}
VALID_SPECIALISTS = {1,2,3,4,5,6,7,8,9,10,11}

# ── Defaults for auto-fix ──────────────────────────────────────────
SPECIALIST_FROM_CONDITION = {
    'cardiac':1,'renal':2,'gi':3,'hepatic':4,
    'respiratory':5,'anemia':6,'diabetes':7,'infection':10
}


def validate_and_fix(p, fname, source_type):
    """
    Validate one patient JSON against v3 schema.
    Auto-fixes minor issues. Returns (fixed_dict, issues_list).
    source_type: 'matched_pair' or 'discharge_summary'
    """
    issues  = []
    p       = deepcopy(p)

    # ── 1. Demographics ────────────────────────────────────────────
    dem = p.get('demographics', {})
    if not isinstance(dem.get('age'), (int, float)) or dem.get('age', 0) <= 0:
        issues.append('demographics.age missing or invalid')
    if dem.get('gender') not in (0, 1):
        issues.append('demographics.gender must be 0 or 1')

    # ── 2. Lab features — ensure all fields exist ──────────────────
    labs = p.get('lab_features', {})
    mask = p.get('lab_missing_mask', {})
    for field in ALL_LABS:
        if field not in labs:
            labs[field] = None
            issues.append(f'lab_features.{field} missing — added as null')
        if f'{field}_missing' not in mask:
            mask[f'{field}_missing'] = 0 if labs[field] is not None else 1
    p['lab_features']    = labs
    p['lab_missing_mask'] = mask

    # ── 3. Check for null values in non-missing fields ─────────────
    for field in ALL_LABS:
        if mask.get(f'{field}_missing', 1) == 0 and labs.get(field) is None:
            mask[f'{field}_missing'] = 1
            issues.append(f'{field} marked measured but is null — corrected mask to missing=1')

    # ── 4. Platelet sanity check ────────────────────────────────────
    plt = labs.get('platelets')
    if plt is not None and plt < 100:
        labs['platelets'] = plt * 100000
        issues.append(f'platelets {plt} looks like lakhs format — converted to {labs["platelets"]}')

    # ── 5. Comorbidities — ensure all 8 fields ─────────────────────
    com = p.get('comorbidities', {})
    for field in REQUIRED_COMORBIDITIES:
        if field not in com:
            com[field] = 0
            issues.append(f'comorbidities.{field} missing — defaulted to 0')
    p['comorbidities'] = com

    # ── 6. Vitals ──────────────────────────────────────────────────
    vit = p.get('vitals', {})
    for field in ['spo2','bp_systolic','bp_diastolic','pulse',
                  'respiratory_rate','temperature_afebrile']:
        if field not in vit:
            vit[field] = None
            issues.append(f'vitals.{field} missing — added as null')
    if vit.get('temperature_afebrile') is None:
        vit['temperature_afebrile'] = 1
    p['vitals'] = vit

    # ── 7. Treatment ───────────────────────────────────────────────
    trt = p.get('treatment', {})
    for field in ['insulin','antibiotics','analgesics','statins']:
        if field not in trt:
            trt[field] = 0
    p['treatment'] = trt

    # ── 8. History ─────────────────────────────────────────────────
    hst = p.get('history', {})
    for field in ['known_diabetes','alcohol','smoking',
                  'diet_vegetarian','sleep_adequate']:
        if field not in hst:
            hst[field] = 0
    p['history'] = hst

    # ── 9. Cardiac findings ────────────────────────────────────────
    crd = p.get('cardiac_findings', {})
    for field in ['lv_diastolic_dysfunction','ef_normal','murmur']:
        if field not in crd:
            crd[field] = 0
    p['cardiac_findings'] = crd

    # ── 10. Condition scores — all 8 fields ────────────────────────
    cs = p.get('condition_scores', {})
    for field in REQUIRED_CONDITION_SCORES:
        if field not in cs:
            cs[field] = 0
            issues.append(f'condition_scores.{field} missing — defaulted to 0')
    p['condition_scores'] = cs

    # ── 11. Organ involvement — all 6 fields ───────────────────────
    org = p.get('organ_involvement', {})
    for field in REQUIRED_ORGAN_INVOLVEMENT:
        if field not in org:
            org[field] = 0
            issues.append(f'organ_involvement.{field} missing — defaulted to 0')
    p['organ_involvement'] = org

    # ── 12. Severity features ──────────────────────────────────────
    sev = p.get('severity_features', {})
    for field in ['age_risk','multi_organ','hypoalbuminemia']:
        if field not in sev:
            sev[field] = 0
    # Auto-derive if age available
    age = p.get('demographics', {}).get('age', 0)
    sev['age_risk'] = 1 if age >= 60 else 0
    alb = labs.get('albumin')
    if alb is not None and mask.get('albumin_missing', 1) == 0:
        sev['hypoalbuminemia'] = 1 if alb < 3.0 else 0
    sev['multi_organ'] = 1 if sum(org.values()) >= 2 else 0
    p['severity_features'] = sev

    # ── 13. Primary condition ──────────────────────────────────────
    pc = p.get('primary_condition', '')
    if pc not in VALID_PRIMARY_CONDITIONS:
        issues.append(f'primary_condition "{pc}" is invalid')

    # ── 14. Targets ────────────────────────────────────────────────
    tgt = p.get('targets', {})
    if 'risk_level' not in tgt or tgt['risk_level'] not in (1,2,3,4):
        tgt['risk_level'] = 2
        issues.append('targets.risk_level missing or invalid — defaulted to 2')
    if 'specialist' not in tgt or tgt['specialist'] not in VALID_SPECIALISTS:
        tgt['specialist'] = SPECIALIST_FROM_CONDITION.get(pc, 10)
        issues.append(f'targets.specialist fixed to {tgt["specialist"]} based on primary_condition')
    p['targets'] = tgt

    # ── 15. Text input ─────────────────────────────────────────────
    txt = p.get('text_input', '')
    if not txt or len(txt) < 10:
        issues.append('text_input is empty or very short')

    # ── 16. External features ──────────────────────────────────────
    if 'external_features' not in p:
        p['external_features'] = {'population_dm_prevalence': 0.10}

    # ── 17. Label confidence ───────────────────────────────────────
    if 'label_confidence' not in p:
        p['label_confidence'] = 0.75 if source_type == 'matched_pair' else 0.85

    # ── 18. Source tag ─────────────────────────────────────────────
    p['_source_type'] = source_type
    p['_filename']    = fname

    return p, issues


def validate_directory(data_dir, source_type, output_dir=None, verbose=True):
    """
    Validate all JSONs in a directory.
    Returns (valid_patients, all_issues_dict)
    """
    data_dir   = Path(data_dir)
    json_files = sorted(data_dir.glob('*.json'))
    json_files = [f for f in json_files if not f.name.startswith('_')]

    valid_patients = []
    all_issues     = {}
    error_count    = 0
    fixed_count    = 0

    if verbose:
        print(f"\nValidating {len(json_files)} files from {data_dir.name}/")
        print("-" * 50)

    for fpath in json_files:
        try:
            p = json.loads(fpath.read_text(encoding='utf-8', errors='ignore'))
        except json.JSONDecodeError as e:
            all_issues[fpath.name] = [f'JSON parse error: {e}']
            error_count += 1
            continue

        fixed, issues = validate_and_fix(p, fpath.name, source_type)

        if issues:
            all_issues[fpath.name] = issues
            fixed_count += 1

        valid_patients.append(fixed)

        # Save fixed version if output_dir provided
        if output_dir:
            out = Path(output_dir) / fpath.name
            out.write_text(json.dumps(fixed, indent=2, ensure_ascii=False))

    if verbose:
        print(f"  Files processed:    {len(json_files)}")
        print(f"  Files with issues:  {fixed_count} (auto-fixed)")
        print(f"  Parse errors:       {error_count}")
        print(f"  Valid patients:     {len(valid_patients)}")

    return valid_patients, all_issues


if __name__ == '__main__':
    import sys
    data_path   = sys.argv[1] if len(sys.argv) > 1 else './data/matched_pairs'
    source_type = sys.argv[2] if len(sys.argv) > 2 else 'matched_pair'
    patients, issues = validate_directory(data_path, source_type)
    if issues:
        print(f"\nIssues summary (first 5):")
        for fname, iss in list(issues.items())[:5]:
            print(f"  {fname}: {iss[0]}")
    print(f"\nDone. {len(patients)} valid patients.")
