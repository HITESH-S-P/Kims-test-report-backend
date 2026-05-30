"""
verify_and_narrate.py
=====================
Orchestrates the full Phase 3 pipeline:
  Component 1 ensemble predictions
  → Mistral verification
  → Mistral narrative
  → Final structured output

HOW TO RUN (local Ollama):
    python verify_and_narrate.py --patient path/to/patient.json

HOW TO RUN (via ngrok):
    python verify_and_narrate.py --patient path/to/patient.json \
        --ngrok https://abc123.ngrok-free.app
"""

import json, argparse, time
import torch
import numpy as np
from pathlib import Path

from features import (extract_features, get_feature_dim,
                      CONDITION_TO_IDX, IDX_TO_CONDITION,
                      IDX_TO_SPECIALIST, ORGAN_FIELDS)
from model import Component1
from mistral_client import get_client
from prompts import build_verification_prompt, build_narrative_prompt, SPECIALIST_NAMES

# ── Paths ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent  # adjust as needed
MODEL_DIR = BASE_DIR / "models" / "component1"
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DROPOUT   = 0.4


# ── Load ensemble models ───────────────────────────────────────────
def load_ensemble(model_dir: Path, input_dim: int):
    """Load all 5 fold models for ensemble inference."""
    cfg_path = model_dir / 'ensemble_config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f"ensemble_config.json not found in {model_dir}")

    cfg    = json.loads(cfg_path.read_text(encoding='utf-8'))
    models = []
    for fpath in cfg['fold_model_paths']:
        checkpoint = torch.load(fpath, map_location=DEVICE, weights_only=True)
        m = Component1(input_dim=input_dim,
                       dropout=cfg.get('dropout', DROPOUT)).to(DEVICE)
        m.load_state_dict(checkpoint['model_state_dict'])
        m.eval()
        models.append(m)

    print(f"✅ Loaded {len(models)} fold models for ensemble")
    return models


# ── Ensemble predict ───────────────────────────────────────────────
def ensemble_predict(models, features_tensor):
    """Average softmax/sigmoid outputs across all fold models."""
    cond_probs_list = []
    risk_probs_list = []
    spec_probs_list = []
    organ_probs_list= []

    with torch.no_grad():
        for m in models:
            out = m.predict(features_tensor)
            cond_probs_list.append(out['condition_probs'].cpu().numpy())
            risk_probs_list.append(out['risk_probs'].cpu().numpy())
            spec_probs_list.append(out['specialist_probs'].cpu().numpy())
            organ_probs_list.append(out['organ_probs'].cpu().numpy())

    cond_probs  = np.mean(cond_probs_list,  axis=0)[0]
    risk_probs  = np.mean(risk_probs_list,  axis=0)[0]
    spec_probs  = np.mean(spec_probs_list,  axis=0)[0]
    organ_probs = np.mean(organ_probs_list, axis=0)[0]

    cond_idx = int(np.argmax(cond_probs))
    risk_idx = int(np.argmax(risk_probs))
    spec_idx = int(np.argmax(spec_probs))
    organ_binary = (organ_probs > 0.5).astype(int).tolist()

    # Confidence = max softmax probability for condition
    cond_confidence = float(cond_probs[cond_idx])

    return {
        'condition_idx':        cond_idx,
        'condition_confidence': cond_confidence,
        'condition_probs':      cond_probs.tolist(),
        'risk_idx':             risk_idx,
        'risk_probs':           risk_probs.tolist(),
        'specialist_idx':       spec_idx,
        'organ_binary':         organ_binary,
        'organ_probs':          organ_probs.tolist(),
    }


# ── Format final output ────────────────────────────────────────────
def format_output(patient, c1_preds, verification, narrative):
    """Assemble the complete structured + narrative output."""
    dem    = patient.get('demographics', {})
    age    = dem.get('age', 0)
    gender = 'Male' if dem.get('gender', 1) == 1 else 'Female'

    cond_name  = verification.get('corrected_condition',
                  IDX_TO_CONDITION.get(c1_preds['condition_idx'], 'unknown')).upper()
    risk_lvl   = verification.get('corrected_risk_level',
                  c1_preds['risk_idx'] + 1)
    spec_name  = verification.get('corrected_specialist',
                  SPECIALIST_NAMES.get(
                      list(IDX_TO_SPECIALIST.keys())[c1_preds['specialist_idx']]
                      if c1_preds['specialist_idx'] < 10 else 10,
                      'General Medicine'))

    organ_binary = c1_preds['organ_binary']
    organs       = [ORGAN_FIELDS[i].upper() for i, v in enumerate(organ_binary) if v == 1]
    extra_organs = [o.upper() for o in verification.get('additional_organ_involvement', [])]
    all_organs   = list(set(organs + extra_organs))

    # FIX 1: Severity flags only include the three structured severity fields.
    # Critical flags from Mistral are kept in their own separate section.
    sev = patient.get('severity_features', {})
    sev_flags = []
    if sev.get('age_risk'):        sev_flags.append('Age Risk (≥60)')
    if sev.get('multi_organ'):     sev_flags.append('Multi-Organ Involvement')
    if sev.get('hypoalbuminemia'): sev_flags.append('Hypoalbuminaemia')

    # Critical flags are NOT merged into sev_flags — displayed separately below
    critical_flags = verification.get('critical_flags', [])

    risk_labels = {1: '1 — Low', 2: '2 — Moderate', 3: '3 — High', 4: '4 — Critical'}
    conf_pct    = f"{c1_preds['condition_confidence']:.0%}"
    verified    = "✅ Verified" if verification.get('predictions_correct', True) else "⚠️ Corrected by AI"

    separator = "═" * 55

    output = f"""
{separator}
  KIMS CLINICAL ANALYSIS
  Patient: {age:.0f}y {gender}
{separator}

RISK LEVEL:          {risk_labels.get(risk_lvl, str(risk_lvl))}
PRIMARY CONDITION:   {cond_name}  ({conf_pct} confidence)  {verified}
RECOMMENDED:         {spec_name}

ORGAN INVOLVEMENT:
  {chr(10).join(f'  ✓ {o}' for o in all_organs) if all_organs else '  None identified'}

SEVERITY FLAGS:
  {chr(10).join(f'  ⚠ {f}' for f in sev_flags) if sev_flags else '  None'}

CRITICAL FLAGS:
  {chr(10).join(f'  ⚠ {f}' for f in critical_flags) if critical_flags else '  None'}

MISSED FINDINGS:
  {chr(10).join(f'  → {f}' for f in verification.get('missed_findings',[])) if verification.get('missed_findings') else '  None'}

{separator}
CLINICAL SUMMARY:
{separator}

{narrative}

{separator}
"""
    return output


# ── Main pipeline ──────────────────────────────────────────────────
def analyse_patient(patient_json_path: str, ngrok_url: str = None,
                    verbose: bool = True) -> dict:
    """
    Full pipeline: patient JSON → structured predictions + narrative.

    Returns dict with:
        structured:  Component 1 predictions (verified)
        narrative:   Clinical summary text
        output_text: Formatted full output
        timings:     Time taken per stage
    """
    t_start = time.time()

    # ── Load patient ───────────────────────────────────────────────
    patient = json.loads(Path(patient_json_path).read_text(encoding='utf-8'))
    if verbose: print(f"Patient loaded: {Path(patient_json_path).name}")

    # ── Component 1 ensemble predict ──────────────────────────────
    input_dim = get_feature_dim()
    models    = load_ensemble(MODEL_DIR, input_dim)
    features  = torch.tensor(
        extract_features(patient), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    t_c1 = time.time()
    c1_preds = ensemble_predict(models, features)
    t_c1_done = time.time()

    if verbose:
        cond = IDX_TO_CONDITION[c1_preds['condition_idx']]
        risk = c1_preds['risk_idx'] + 1
        print(f"Component 1: {cond} | risk {risk} | "
              f"conf {c1_preds['condition_confidence']:.0%} "
              f"({t_c1_done - t_c1:.2f}s)")

    # ── Mistral verification ───────────────────────────────────────
    client = get_client(ngrok_url)
    t_v = time.time()

    verification_prompt = build_verification_prompt(patient, c1_preds)
    try:
        verification = client.generate_json(verification_prompt)
        if verbose:
            correct = verification.get('predictions_correct', True)
            print(f"Verification: {'✅ correct' if correct else '⚠️ corrected'} "
                  f"({time.time()-t_v:.1f}s)")
    except Exception as e:
        if verbose: print(f"⚠️  Verification failed ({e}), using C1 predictions as-is")
        verification = {
            'predictions_correct': True,
            'corrected_condition':  IDX_TO_CONDITION[c1_preds['condition_idx']],
            'corrected_risk_level': c1_preds['risk_idx'] + 1,
            'corrected_specialist': SPECIALIST_NAMES.get(
                list(IDX_TO_SPECIALIST.keys())[c1_preds['specialist_idx']]
                if c1_preds['specialist_idx'] < 10 else 10, 'General Medicine'),
            'additional_organ_involvement': [],
            'missed_findings': [],
            'critical_flags': [],
            'confidence_note': f'Verification unavailable: {e}',
        }

    # ── Mistral narrative ──────────────────────────────────────────
    t_n = time.time()

    # FIX 2: risk level line now explicitly tells Mistral the exact phrasing to use,
    # preventing it from substituting synonyms like "moderate risk" for risk level 3.
    narrative_prompt = build_narrative_prompt(patient, c1_preds, verification)
    narrative = client.generate(narrative_prompt, temperature=0.3, max_tokens=600)
    if verbose: print(f"Narrative generated ({time.time()-t_n:.1f}s)")

    # ── Format output ──────────────────────────────────────────────
    output_text = format_output(patient, c1_preds, verification, narrative)
    t_total = time.time() - t_start

    result = {
        'structured': {
            'primary_condition': verification.get('corrected_condition',
                IDX_TO_CONDITION[c1_preds['condition_idx']]),
            'risk_level':        verification.get('corrected_risk_level',
                c1_preds['risk_idx'] + 1),
            'specialist':        verification.get('corrected_specialist'),
            'organ_involvement': c1_preds['organ_binary'],
            'condition_confidence': c1_preds['condition_confidence'],
            'predictions_verified': verification.get('predictions_correct', True),
            'missed_findings':   verification.get('missed_findings', []),
            'critical_flags':    verification.get('critical_flags', []),
        },
        'narrative':    narrative,
        'output_text':  output_text,
        'timings': {
            'component1_s':   round(t_c1_done - t_c1, 2),
            'verification_s': round(time.time() - t_v, 2),
            'narrative_s':    round(time.time() - t_n, 2),
            'total_s':        round(t_total, 2),
        }
    }

    if verbose:
        print(output_text)
        print(f"Total time: {t_total:.1f}s")

    return result


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KIMS Clinical Analysis Pipeline')
    parser.add_argument('--patient', required=True, help='Path to patient JSON file')
    parser.add_argument('--ngrok',   default=None,  help='ngrok URL for Mistral (optional)')
    parser.add_argument('--save',    default=None,  help='Save output to this JSON file')
    args = parser.parse_args()

    result = analyse_patient(args.patient, ngrok_url=args.ngrok)

    if args.save:
        Path(args.save).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(f"\nOutput saved to {args.save}")