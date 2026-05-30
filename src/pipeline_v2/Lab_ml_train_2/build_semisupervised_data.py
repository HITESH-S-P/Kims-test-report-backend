r"""
build_semisupervised_data.py
============================
Turns the 2000 parsed unlabelled reports into usable training data.

Two distinct outputs, each with different label provenance:

  1. NORMAL cases (deterministic, high trust)
     A report is "normal" if EVERY measured core lab falls within its
     reference range AND no critical thresholds are crossed.
     These get is_healthy=1, used to train the binary health head.
     Label weight: 0.85 (high — deterministic rule, not a guess)

  2. PSEUDO-LABELLED condition cases (model-guessed, lower trust)
     For reports that are clearly NOT normal (have real abnormalities),
     run the current Component-1 ensemble. Keep only predictions with
     confidence > 0.85. These get the predicted condition as label.
     Label weight: 0.60 (lower — model guess, may be wrong)
     is_healthy=0.

Reports that are neither clean-normal nor high-confidence are DISCARDED
(we never train on ambiguous guesses — medical scope of error must stay low).

Output: a folder of v3-schema training JSONs ready for train_component1_v2.py
"""

import json, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch

# Reference ranges (our hardcoded source of truth)
from lab_ranges import (REFERENCE_RANGES, get_range, classify_value,
                        is_critical, deviation_severity)

# Component 1 (for pseudo-labelling) — these live in component1 dir
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
sys.path.insert(0, str(PROJECT_ROOT / "src" / "component1"))

# ── Config ─────────────────────────────────────────────────────────
PSEUDO_CONF_THRESHOLD = 0.85    # balanced — per your choice
NORMAL_WEIGHT         = 0.85    # deterministic label, high trust
PSEUDO_WEIGHT         = 0.60    # model guess, lower trust

# A report needs at least this many core labs measured to be judged
MIN_CORE_LABS_FOR_NORMAL = 8

# Core labs that matter most for the "is this person healthy" decision
CORE_HEALTH_LABS = [
    'hemoglobin', 'platelets', 'wbc', 'creatinine', 'urea',
    'sodium', 'potassium', 'bilirubin_total', 'albumin',
    'sgot', 'sgpt', 'alp', 'hba1c', 'grbs',
]

CORE_LABS = ['hemoglobin','platelets','wbc','esr','bilirubin_total',
             'albumin','ferritin','serum_iron','creatinine','urea',
             'grbs','bmi','hba1c','tsat','ef_percent']
V3_EXTRA_LABS = ['sodium','potassium','chloride','sgot','sgpt','alp','ggt',
                 'crp','uric_acid','pcv','mcv','rdw',
                 'neutrophils_pct','lymphocytes_pct']
ALL_LABS = CORE_LABS + V3_EXTRA_LABS


# ── Normality judgement ────────────────────────────────────────────
def judge_normality(labs, printed_ranges, gender_str='female'):
    """
    Decide if a patient is clinically normal using BOTH:
      - printed ranges from their own report (preferred)
      - our hardcoded ranges (fallback)

    Returns dict:
      {is_normal, n_measured, n_abnormal, abnormal_fields,
       abnormality_score, any_critical}
    """
    measured        = 0
    abnormal        = 0
    abnormal_fields = []
    total_severity  = 0.0
    any_critical    = False

    for field in CORE_HEALTH_LABS:
        val = labs.get(field)
        if val is None:
            continue
        measured += 1

        # Prefer the report's own printed range; fall back to hardcoded
        rng = None
        if field in printed_ranges:
            rng = tuple(printed_ranges[field])
        else:
            rng = get_range(field, gender_str)

        # Critical check (absolute thresholds, always)
        crit, _ = is_critical(field, val)
        if crit:
            any_critical = True
            abnormal += 1
            abnormal_fields.append(field)
            total_severity += 2.0
            continue

        if rng:
            lo, hi = rng
            if val < lo or val > hi:
                abnormal += 1
                abnormal_fields.append(field)
                total_severity += deviation_severity(field, val, gender_str)

    # Decision: normal if enough labs measured AND none abnormal AND no critical
    is_normal = (measured >= MIN_CORE_LABS_FOR_NORMAL
                 and abnormal == 0
                 and not any_critical)

    return {
        'is_normal':         is_normal,
        'n_measured':        measured,
        'n_abnormal':        abnormal,
        'abnormal_fields':   abnormal_fields,
        'abnormality_score': round(total_severity, 2),
        'any_critical':      any_critical,
    }


# ── Build a v3-schema patient JSON from raw labs ───────────────────
def build_v3_json(patient_id, labs, primary_condition, is_healthy,
                  label_weight, label_source, gender=0):
    """Assemble a minimal but complete v3-schema training JSON."""
    lab_features = {}
    lab_missing  = {}
    for f in ALL_LABS:
        v = labs.get(f)
        lab_features[f]             = v
        lab_missing[f'{f}_missing'] = 0 if v is not None else 1

    # Derive organ involvement + severity from labs (rule-based)
    org = {
        'hepatic':     1 if (labs.get('sgot') and labs['sgot'] > 80) else 0,
        'respiratory': 0,
        'cardiac':     0,
        'renal':       1 if (labs.get('creatinine') and labs['creatinine'] > 1.2) else 0,
        'gi':          0,
        'urological':  0,
    }
    alb = labs.get('albumin')
    sev = {
        'age_risk':        0,
        'multi_organ':     1 if sum(org.values()) >= 2 else 0,
        'hypoalbuminemia': 1 if (alb and alb < 3.0) else 0,
    }

    # Condition scores (zeros for normal, simple derivation otherwise)
    scores = {k: 0 for k in ['infection','anemia','gi','diabetes',
                              'respiratory','cardiac','renal','hepatic']}
    if not is_healthy and primary_condition in scores:
        scores[primary_condition] = 3

    return {
        'text_input':       '',
        'demographics':     {'age': 45.0, 'gender': int(gender)},
        'lab_features':     lab_features,
        'lab_missing_mask': lab_missing,
        'comorbidities':    {k: 0 for k in [
            'diabetes','hypertension','cardiac_disease','respiratory_disease',
            'renal_disease','hepatic_disease','urological_condition',
            'neurological_condition']},
        'vitals':           {'spo2': None, 'bp_systolic': None,
                             'bp_diastolic': None, 'pulse': None,
                             'respiratory_rate': None, 'temperature_afebrile': 1},
        'treatment':        {'insulin': 0, 'antibiotics': 0,
                             'analgesics': 0, 'statins': 0},
        'history':          {k: 0 for k in [
            'known_diabetes','alcohol','smoking','diet_vegetarian','sleep_adequate']},
        'cardiac_findings': {'lv_diastolic_dysfunction': 0, 'ef_normal': 1, 'murmur': 0},
        'condition_scores': scores,
        'organ_involvement': org,
        'severity_features': sev,
        'primary_condition': primary_condition if not is_healthy else 'normal',
        'external_features': {'population_dm_prevalence': 0.10},
        'label_confidence':  label_weight,
        'is_healthy':        int(is_healthy),
        '_label_source':     label_source,
        '_patient_id':       patient_id,
        'targets':           {'risk_level': 1 if is_healthy else 2,
                              'specialist': 10},
    }


# ── Main builder ───────────────────────────────────────────────────
def build(parsed_dir, out_dir, model_dir, use_rich=True):
    parsed_dir = Path(parsed_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports (need the model for pseudo-labelling)
    from features import extract_features, get_feature_dim, IDX_TO_CONDITION
    from model import Component1

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ensemble for pseudo-labelling
    cfg = json.loads((Path(model_dir) / 'ensemble_config.json').read_text())
    input_dim = cfg['input_dim']
    models = []
    for fp in cfg['fold_model_paths']:
        ck = torch.load(fp, map_location=DEVICE, weights_only=True)
        m  = Component1(input_dim=input_dim, dropout=cfg.get('dropout', 0.4)).to(DEVICE)
        m.load_state_dict(ck['model_state_dict'])
        m.eval()
        models.append(m)

    def ensemble_predict(patient_json):
        feats = torch.tensor(extract_features(patient_json),
                             dtype=torch.float32).unsqueeze(0).to(DEVICE)
        probs = []
        with torch.no_grad():
            for m in models:
                out = m.predict(feats)
                probs.append(out['condition_probs'].cpu().numpy())
        avg = np.mean(probs, axis=0)[0]
        idx = int(np.argmax(avg))
        return IDX_TO_CONDITION[idx], float(avg[idx])

    # Stats
    stats = {
        'total': 0, 'normal': 0, 'pseudo_kept': 0,
        'discarded_lowconf': 0, 'discarded_few_labs': 0,
        'pseudo_by_condition': Counter(),
    }

    files = sorted(parsed_dir.glob('*.json'))
    files = [f for f in files if not f.name.startswith('_')]

    # ── Optional Rich progress ─────────────────────────────────────
    if use_rich:
        try:
            from rich.progress import (Progress, BarColumn, TextColumn,
                                       MofNCompleteColumn, TimeElapsedColumn)
            from rich.console import Console
            console = Console()
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=console,
            )
        except ImportError:
            use_rich = False

    def process_all():
        for fpath in files:
            stats['total'] += 1
            rec = json.loads(fpath.read_text(encoding='utf-8'))
            labs   = rec['lab_features']
            ranges = rec.get('lab_printed_ranges', {})
            pid    = rec['patient_id']

            # Step 1: normality judgement
            judge = judge_normality(labs, ranges)

            if judge['n_measured'] < MIN_CORE_LABS_FOR_NORMAL:
                stats['discarded_few_labs'] += 1
                # Still may pseudo-label if it has some labs, but skip if too sparse
                if judge['n_measured'] < 5:
                    continue

            if judge['is_normal']:
                # NORMAL case — deterministic, high trust
                pj = build_v3_json(pid, labs, 'normal', is_healthy=1,
                                   label_weight=NORMAL_WEIGHT,
                                   label_source='deterministic_normal')
                (out_dir / f"normal_{pid}.json").write_text(
                    json.dumps(pj, indent=2), encoding='utf-8')
                stats['normal'] += 1
            else:
                # NOT normal → pseudo-label with ensemble
                tmp = build_v3_json(pid, labs, 'infection', is_healthy=0,
                                    label_weight=PSEUDO_WEIGHT,
                                    label_source='pseudo')
                cond, conf = ensemble_predict(tmp)
                if conf >= PSEUDO_CONF_THRESHOLD:
                    pj = build_v3_json(pid, labs, cond, is_healthy=0,
                                       label_weight=PSEUDO_WEIGHT,
                                       label_source=f'pseudo_conf{conf:.2f}')
                    (out_dir / f"pseudo_{cond}_{pid}.json").write_text(
                        json.dumps(pj, indent=2), encoding='utf-8')
                    stats['pseudo_kept'] += 1
                    stats['pseudo_by_condition'][cond] += 1
                else:
                    stats['discarded_lowconf'] += 1

            if use_rich:
                progress.update(task, advance=1)

    if use_rich:
        with progress:
            task = progress.add_task("Building semi-supervised data", total=len(files))
            process_all()
    else:
        process_all()

    # Summary
    stats['pseudo_by_condition'] = dict(stats['pseudo_by_condition'])
    (out_dir / '_build_summary.json').write_text(
        json.dumps(stats, indent=2), encoding='utf-8')

    # Print summary
    print(f"\n{'='*55}")
    print(f"  SEMI-SUPERVISED DATA BUILD COMPLETE")
    print(f"{'='*55}")
    print(f"  Total reports processed:   {stats['total']}")
    print(f"  Normal cases (det.):       {stats['normal']}")
    print(f"  Pseudo-labelled (>{PSEUDO_CONF_THRESHOLD}):   {stats['pseudo_kept']}")
    print(f"  Discarded (low conf):      {stats['discarded_lowconf']}")
    print(f"  Discarded (too few labs):  {stats['discarded_few_labs']}")
    print(f"\n  Pseudo-label distribution:")
    for cond, n in sorted(stats['pseudo_by_condition'].items(), key=lambda x: -x[1]):
        print(f"    {cond:<14} {n}")
    print(f"\n  Output: {out_dir}")
    return stats


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        build(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python build_semisupervised_data.py "
              "<parsed_dir> <out_dir> <model_dir>")
        print("\nExample:")
        print(r'  python build_semisupervised_data.py \\')
