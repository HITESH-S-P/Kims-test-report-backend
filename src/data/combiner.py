"""
combiner.py
===========
Merges matched_pairs + discharge_summaries into one
combined training dataset with class balancing.

Run:
    python combiner.py

Output: data/combined/ — final training JSONs ready for Component 1 + 2
"""

import json, random, re
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict
from validator import validate_directory

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent.parent  # kims_v3/
MATCHED_DIR  = ROOT / 'data' / 'matched_pairs'
DISCHARGE_DIR= ROOT / 'data' / 'discharge_summaries'
COMBINED_DIR = ROOT / 'data' / 'combined'
COMBINED_DIR.mkdir(exist_ok=True)

# ── Class balance target ───────────────────────────────────────────
# Aim for at least this many examples per class after augmentation
MIN_PER_CLASS = 65

# ── Text augmentation (same as v2 but expanded) ────────────────────
SUBSTITUTIONS = [
    ('breathlessness', 'dyspnoea'),
    ('dyspnoea', 'shortness of breath'),
    ('gradually progressive', 'progressively worsening'),
    ('insidious in onset', 'gradual in onset'),
    ('apparently alright', 'apparently well'),
    ('apparently well', 'apparently normal'),
    ('intermittent', 'episodic'),
    ('associated with', 'accompanied by'),
    ('burning micturition', 'dysuria'),
    ('fever with chills', 'febrile illness with rigors'),
    ('reduced appetite', 'decreased appetite'),
    ('weight loss', 'loss of weight'),
    ('generalised weakness', 'generalised fatigue'),
    ('loose stools', 'diarrhoea'),
    ('vomiting', 'emesis'),
    ('chest pain', 'chest discomfort'),
    ('since one month', 'over the past month'),
    ('since 1 month', 'for approximately one month'),
    ('nausea and vomiting', 'nausea with emesis'),
    ('loss of appetite', 'anorexia'),
]


def augment_patient(p):
    """
    Light augmentation — text paraphrase + ±5% numeric perturbation.
    Used only for minority class oversampling.
    """
    aug  = deepcopy(p)
    text = aug.get('text_input', '')
    chosen = random.sample(SUBSTITUTIONS, min(3, len(SUBSTITUTIONS)))
    for src, tgt in chosen:
        if src.lower() in text.lower():
            text = re.sub(re.escape(src), tgt, text, count=1, flags=re.IGNORECASE)
    aug['text_input'] = text

    # Numeric perturbation on measured labs only
    labs = aug.get('lab_features', {})
    mask = aug.get('lab_missing_mask', {})
    perturb_fields = ['hemoglobin','wbc','creatinine','urea','albumin',
                      'sgot','sgpt','sodium','potassium']
    for field in perturb_fields:
        if mask.get(f'{field}_missing', 1) == 0 and labs.get(field) is not None:
            noise = random.uniform(-0.04, 0.04)
            labs[field] = round(labs[field] * (1 + noise), 2)
    aug['lab_features']  = labs
    aug['_augmented']    = True
    aug['_aug_source']   = p.get('_filename', '')
    return aug


def merge_discharge_into_v3_schema(p):
    """
    Convert a discharge-summary-only JSON (v2 schema) into v3 schema.
    Adds v3 extra lab fields as missing, preserves everything else.
    """
    from validator import ALL_LABS, validate_and_fix
    fixed, _ = validate_and_fix(p, p.get('_filename', 'unknown'), 'discharge_summary')
    return fixed


def combine_and_balance():
    """Main function — combine, validate, balance, write output."""

    # ── Step 1: Validate matched pairs ────────────────────────────
    print("=" * 60)
    print("  KIMS V3 — DATA COMBINER")
    print("=" * 60)

    matched, mp_issues = validate_directory(
        MATCHED_DIR, 'matched_pair', verbose=True
    )
    print(f"  Issues auto-fixed: {len(mp_issues)}")

    # ── Step 2: Validate discharge summaries ──────────────────────
    discharge, ds_issues = validate_directory(
        DISCHARGE_DIR, 'discharge_summary', verbose=True
    )
    print(f"  Issues auto-fixed: {len(ds_issues)}")

    # ── Step 3: Combine ────────────────────────────────────────────
    # Matched pairs get higher label_confidence (real labs)
    # Discharge summaries may have synthesized labs (lower confidence)
    all_patients = matched + discharge

    print(f"\nCombined total: {len(all_patients)} patients")
    dist = Counter(p['primary_condition'] for p in all_patients
                   if p.get('primary_condition') in {
                       'infection','anemia','gi','diabetes',
                       'respiratory','cardiac','renal','hepatic'})
    print("\nClass distribution before balancing:")
    for cond, count in sorted(dist.items(), key=lambda x: -x[1]):
        bar = '█' * count + '░' * max(0, MIN_PER_CLASS - count)
        print(f"  {cond:<15} {count:>3}  {bar[:50]}")

    # ── Step 4: Oversample minority classes ───────────────────────
    by_class = defaultdict(list)
    for p in all_patients:
        pc = p.get('primary_condition', '')
        if pc in dist:
            by_class[pc].append(p)

    augmented = list(all_patients)
    aug_log   = {}
    for cond, samples in by_class.items():
        n      = len(samples)
        needed = max(0, MIN_PER_CLASS - n)
        aug_log[cond] = {'original': n, 'added': needed}
        # Prefer matched_pair samples for augmentation (real labs)
        mp_samples = [s for s in samples if s.get('_source_type') == 'matched_pair']
        pool       = mp_samples if mp_samples else samples
        for _ in range(needed):
            augmented.append(augment_patient(random.choice(pool)))

    random.shuffle(augmented)

    print(f"\nClass distribution after balancing:")
    dist2 = Counter(p['primary_condition'] for p in augmented
                    if p.get('primary_condition') in dist)
    for cond, count in sorted(dist2.items(), key=lambda x: -x[1]):
        orig = aug_log.get(cond, {}).get('original', count)
        added = aug_log.get(cond, {}).get('added', 0)
        bar = '█' * orig + '░' * added
        print(f"  {cond:<15} {orig:>3} + {added:>2} aug = {count:>3}  {bar[:55]}")

    # ── Step 5: Write combined dataset ────────────────────────────
    written = 0
    for i, p in enumerate(augmented):
        fname = p.get('_filename', f'patient_{i:04d}.json')
        if p.get('_augmented'):
            fname = fname.replace('.json', f'_aug{i}.json')
        fname = fname.replace(' ', '_')
        out   = COMBINED_DIR / fname
        out.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding='utf-8')
        written += 1

    # ── Step 6: Write dataset manifest ────────────────────────────
    manifest = {
        'total': written,
        'matched_pairs': len(matched),
        'discharge_summaries': len(discharge),
        'augmented_added': sum(v['added'] for v in aug_log.values()),
        'class_distribution': dict(dist2),
        'min_per_class': MIN_PER_CLASS,
    }
    (COMBINED_DIR / '_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {COMBINED_DIR}/")
    print(f"  Total training files: {written}")
    print(f"    Matched pairs:       {len(matched)}")
    print(f"    Discharge summaries: {len(discharge)}")
    print(f"    Augmented:           {sum(v['added'] for v in aug_log.values())}")
    print(f"{'='*60}")
    print(f"\n✅ Phase 1 complete. Run Component 1 training next.")

    return augmented, manifest


if __name__ == '__main__':
    combine_and_balance()
