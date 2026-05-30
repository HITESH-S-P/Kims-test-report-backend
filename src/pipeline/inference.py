"""
inference.py
============
Full end-to-end inference pipeline for the KIMS Clinical AI.

Input:  patient JSON dict (built by rule_scorer.build_patient_json)
Output: structured result dict with predictions, narrative, timings

This module is intentionally stateless — models are loaded once and
passed in, not reloaded on every call. Call load_models() at startup.

Usage:
    from inference import load_models, load_models_v2, run_inference

    models = load_models()                          # call once at startup
    models['v2_models'] = load_models_v2()          # call once at startup
    result = run_inference(patient_json, models)    # call per patient
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import sys

# Add component1 + component2 to path so model.py and features.py are importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust as needed

SRC_C1  = PROJECT_ROOT / "src" / "component1"
SRC_C2  = PROJECT_ROOT / "src" / "component2"
SRC_PV2 = PROJECT_ROOT / "src" / "pipeline_v2" / "Lab_ml_train_2"

for p in [str(SRC_C1), str(SRC_C2), str(SRC_PV2)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Paths ──────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR    = PROJECT_ROOT / "models" / "component1"
DEFAULT_MODEL_DIR_V2 = PROJECT_ROOT / "models" / "component1_v2"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Label maps (must match training) ──────────────────────────────
CONDITION_NAMES = {
    0: 'infection', 1: 'anemia', 2: 'gi', 3: 'diabetes',
    4: 'respiratory', 5: 'cardiac', 6: 'renal', 7: 'hepatic',
}
ORGAN_FIELDS = ['hepatic', 'respiratory', 'cardiac', 'renal', 'gi', 'urological']

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

RISK_LABELS = {
    1: '1 — Low',
    2: '2 — Moderate',
    3: '3 — High',
    4: '4 — Critical',
}

RISK_COLORS = {
    1: 'green',
    2: 'orange',
    3: 'red',
    4: 'darkred',
}


# ══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════

def load_models(model_dir: str = None) -> dict:
    """
    Load all 5 fold v1 models for ensemble inference.
    Call once at Streamlit startup (cached with st.cache_resource).

    Returns dict with:
        'models':     list of Component1 model objects
        'device':     torch.device
        'model_dir':  Path
        'status':     'ok' | 'error'
        'message':    error message if status == 'error'
    """
    from features import get_feature_dim
    from model import Component1

    mdir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    cfg_path = mdir / 'ensemble_config.json'

    if not cfg_path.exists():
        return {
            'models': [],
            'device': DEVICE,
            'model_dir': mdir,
            'status': 'error',
            'message': f'ensemble_config.json not found in {mdir}',
        }

    try:
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        input_dim = get_feature_dim()
        dropout = cfg.get('dropout', 0.4)

        models = []
        for fpath in cfg['fold_model_paths']:
            checkpoint = torch.load(fpath, map_location=DEVICE, weights_only=True)
            m = Component1(input_dim=input_dim, dropout=dropout).to(DEVICE)
            m.load_state_dict(checkpoint['model_state_dict'])
            m.eval()
            models.append(m)

        return {
            'models':    models,
            'device':    DEVICE,
            'model_dir': mdir,
            'status':    'ok',
            'message':   f'Loaded {len(models)} fold models from {mdir}',
        }

    except Exception as e:
        return {
            'models': [],
            'device': DEVICE,
            'model_dir': mdir,
            'status':  'error',
            'message': str(e),
        }


def load_models_v2(model_dir: str = None) -> dict:
    """
    Load all 5 fold v2 models (with health head) for gate inference.
    Call once at Streamlit startup alongside load_models().

    Returns dict with same structure as load_models().
    """
    # SRC_PV2 is already on sys.path from module-level insert above
    from features import get_feature_dim
    from model_v2 import Component1V2

    mdir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR_V2
    cfg_path = mdir / 'ensemble_config.json'

    if not cfg_path.exists():
        return {
            'models': [],
            'device': DEVICE,
            'model_dir': mdir,
            'status': 'error',
            'message': f'ensemble_config.json not found in {mdir}',
        }

    try:
        cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
        input_dim = get_feature_dim()
        dropout = cfg.get('dropout', 0.4)

        models = []
        for fpath in cfg['fold_model_paths']:
            checkpoint = torch.load(fpath, map_location=DEVICE, weights_only=True)
            m = Component1V2(input_dim=input_dim, dropout=dropout).to(DEVICE)
            m.load_state_dict(checkpoint['model_state_dict'])
            m.eval()
            models.append(m)

        return {
            'models':    models,
            'device':    DEVICE,
            'model_dir': mdir,
            'status':    'ok',
            'message':   f'Loaded {len(models)} v2 fold models from {mdir}',
        }

    except Exception as e:
        return {
            'models': [],
            'device': DEVICE,
            'model_dir': mdir,
            'status':  'error',
            'message': str(e),
        }


# ══════════════════════════════════════════════════════════════════
#  COMPONENT 1 ENSEMBLE PREDICT
# ══════════════════════════════════════════════════════════════════

def _ensemble_predict(models: list, features_tensor) -> dict:
    """Average softmax/sigmoid outputs across all fold v1 models."""
    cond_list  = []
    risk_list  = []
    spec_list  = []
    organ_list = []

    with torch.no_grad():
        for m in models:
            out = m.predict(features_tensor)
            cond_list.append(out['condition_probs'].cpu().numpy())
            risk_list.append(out['risk_probs'].cpu().numpy())
            spec_list.append(out['specialist_probs'].cpu().numpy())
            organ_list.append(out['organ_probs'].cpu().numpy())

    cond_probs  = np.mean(cond_list,  axis=0)[0]
    risk_probs  = np.mean(risk_list,  axis=0)[0]
    spec_probs  = np.mean(spec_list,  axis=0)[0]
    organ_probs = np.mean(organ_list, axis=0)[0]

    cond_idx = int(np.argmax(cond_probs))
    risk_idx = int(np.argmax(risk_probs))
    spec_idx = int(np.argmax(spec_probs))
    organ_binary = (organ_probs > 0.5).astype(int).tolist()

    IDX_TO_SPECIALIST = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10}

    return {
        'condition_idx':        cond_idx,
        'condition_name':       CONDITION_NAMES[cond_idx],
        'condition_confidence': float(cond_probs[cond_idx]),
        'condition_probs':      cond_probs.tolist(),
        'risk_idx':             risk_idx,
        'risk_level':           risk_idx + 1,
        'risk_probs':           risk_probs.tolist(),
        'specialist_idx':       spec_idx,
        'specialist_code':      IDX_TO_SPECIALIST.get(spec_idx, 10),
        'organ_binary':         organ_binary,
        'organ_probs':          organ_probs.tolist(),
    }


def _health_head_predict(v2_models: list, features_tensor) -> dict:
    """
    Run the v2 health head across all fold models and average.
    Returns dict with 'health_idx' and 'healthy_confidence'
    that clinical_gate.run_gate() expects.
    """
    health_probs_list = []

    with torch.no_grad():
        for m in v2_models:
            out = m.predict(features_tensor)
            # health_probs shape: (1, 2) — [sick_prob, healthy_prob]
            health_probs_list.append(out['health_probs'].cpu().numpy())

    avg_health = np.mean(health_probs_list, axis=0)[0]  # shape: (2,)
    health_idx = int(np.argmax(avg_health))              # 0=sick, 1=healthy
    healthy_confidence = float(avg_health[1])            # P(healthy)

    return {
        'health_idx':         health_idx,
        'healthy_confidence': healthy_confidence,
    }


# ══════════════════════════════════════════════════════════════════
#  MISTRAL CALLS
# ══════════════════════════════════════════════════════════════════

def _call_mistral_verification(patient: dict, c1_preds: dict,
                                ngrok_url: str = None) -> dict:
    """Run Mistral verification prompt. Returns verification dict."""
    from mistral_client import get_client
    from prompts import build_verification_prompt

    client = get_client(ngrok_url)
    prompt = build_verification_prompt(patient, c1_preds)
    return client.generate_json(prompt)


def _call_mistral_narrative(patient: dict, c1_preds: dict,
                             verification: dict,
                             ngrok_url: str = None) -> str:
    """Run Mistral narrative prompt. Returns narrative string."""
    from mistral_client import get_client
    from prompts import build_narrative_prompt

    client = get_client(ngrok_url)
    prompt = build_narrative_prompt(patient, c1_preds, verification)
    return client.generate(prompt, temperature=0.3, max_tokens=600)


# ══════════════════════════════════════════════════════════════════
#  STRUCTURED OUTPUT BUILDER
# ══════════════════════════════════════════════════════════════════

def _build_structured_output(patient: dict, c1_preds: dict,
                              verification: dict) -> dict:
    """
    Merge C1 predictions + Mistral verification into final structured output.
    """
    dem    = patient.get('demographics', {})
    age    = int(dem.get('age', 0))
    gender = 'Male' if dem.get('gender', 1) == 1 else 'Female'

    condition = verification.get(
        'corrected_condition', c1_preds['condition_name']
    ).lower()
    risk_level = int(verification.get(
        'corrected_risk_level', c1_preds['risk_level']
    ))
    specialist_name = verification.get('corrected_specialist')
    if not specialist_name:
        specialist_name = SPECIALIST_NAMES.get(
            c1_preds['specialist_code'], 'General Medicine'
        )

    organ_binary = c1_preds['organ_binary']
    organs_from_c1 = [
        ORGAN_FIELDS[i].upper()
        for i, v in enumerate(organ_binary) if v == 1
    ]
    extra_organs = [
        o.upper()
        for o in verification.get('additional_organ_involvement', [])
    ]
    all_organs = list(dict.fromkeys(organs_from_c1 + extra_organs))

    sev = patient.get('severity_features', {})
    severity_flags = []
    if sev.get('age_risk'):        severity_flags.append('Age Risk (≥60 years)')
    if sev.get('multi_organ'):     severity_flags.append('Multi-Organ Involvement')
    if sev.get('hypoalbuminemia'): severity_flags.append('Hypoalbuminaemia')

    critical_flags  = verification.get('critical_flags', [])
    missed_findings = verification.get('missed_findings', [])

    return {
        'age':               age,
        'gender':            gender,
        'condition':         condition,
        'condition_upper':   condition.upper(),
        'condition_confidence': c1_preds['condition_confidence'],
        'risk_level':        risk_level,
        'risk_label':        RISK_LABELS.get(risk_level, str(risk_level)),
        'risk_color':        RISK_COLORS.get(risk_level, 'gray'),
        'specialist_name':   specialist_name,
        'organs':            all_organs,
        'severity_flags':    severity_flags,
        'critical_flags':    critical_flags,
        'missed_findings':   missed_findings,
        'predictions_verified': verification.get('predictions_correct', True),
        'verification_note': verification.get('confidence_note', ''),
    }


def _build_tiered_structured_output(patient: dict, tiered_response: dict) -> dict:
    """
    Build a structured output dict for the tiered normal response
    so render_results() in app.py always receives the same shape.
    """
    dem    = patient.get('demographics', {})
    age    = int(dem.get('age', 0))
    gender = 'Male' if dem.get('gender', 1) == 1 else 'Female'

    return {
        'age':               age,
        'gender':            gender,
        'condition':         'no acute concern',
        'condition_upper':   'NO ACUTE CONCERN',
        'condition_confidence': 1.0,
        'risk_level':        1,
        'risk_label':        RISK_LABELS[1],
        'risk_color':        'green',
        'specialist_name':   'General Medicine / Routine Follow-up',
        'organs':            [],
        'severity_flags':    [],
        'critical_flags':    [],
        'missed_findings':   [],
        'predictions_verified': True,
        'verification_note': tiered_response.get('detail', ''),
        # Extra keys for the tiered UI card
        'is_tiered_normal':  True,
        'tiered_tier':       tiered_response.get('tier', 'no_concern'),
        'tiered_headline':   tiered_response.get('headline', ''),
        'tiered_detail':     tiered_response.get('detail', ''),
        'tiered_recommendation': tiered_response.get('recommendation', ''),
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════════

def run_inference(
    patient: dict,
    loaded_models: dict,
    ngrok_url: str = None,
    skip_mistral: bool = False,
) -> dict:
    """
    Full inference pipeline.

    Args:
        patient:       v3 patient JSON dict (from rule_scorer.build_patient_json)
        loaded_models: dict returned by load_models(), with 'v2_models' key added
                       by load_models_v2() (set in app.py get_models())
        ngrok_url:     optional ngrok URL for Mistral (None = localhost)
        skip_mistral:  if True, skip Mistral calls (for testing/offline use)

    Returns dict with:
        'structured':   final predictions dict
        'narrative':    clinical summary text
        'c1_raw':       raw Component 1 predictions (None if gate short-circuits)
        'verification': raw Mistral verification dict
        'timings':      per-stage timing
        'status':       'ok' | 'error'
        'error':        error message if status == 'error'
        'gate_result':  gate dict (always present if gate ran)
    """
    timings = {}
    t_total = time.time()

    # ── Validate models loaded ─────────────────────────────────────
    if loaded_models.get('status') != 'ok' or not loaded_models['models']:
        return {
            'status': 'error',
            'error':  f"Models not loaded: {loaded_models.get('message', 'unknown error')}",
            'structured': None, 'narrative': '', 'c1_raw': None,
            'verification': {}, 'timings': {}, 'gate_result': None,
        }

    # ── Extract features (needed for both gate and C1) ─────────────
    try:
        from features import extract_features
        feats_np = extract_features(patient)
        feats_t  = torch.tensor(feats_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return {
            'status': 'error',
            'error':  f'Feature extraction failed: {e}',
            'structured': None, 'narrative': '', 'c1_raw': None,
            'verification': {}, 'timings': {}, 'gate_result': None,
        }

    # ── Clinical gate ──────────────────────────────────────────────
    # Pulls labs + mask directly from patient dict (what run_gate() expects)
    gate_result = None
    try:
        from clinical_gate import run_gate

        labs = patient.get('lab_features', {})
        mask = patient.get('lab_missing_mask', {})
        dem  = patient.get('demographics', {})
        gender_str = 'male' if dem.get('gender', 1) == 1 else 'female'

        # Run health head (Layer B) if v2 models are available
        health_head_output = None
        v2 = loaded_models.get('v2_models', {})
        if v2.get('status') == 'ok' and v2.get('models'):
            try:
                health_head_output = _health_head_predict(v2['models'], feats_t)
            except Exception as e:
                # Non-fatal — gate will run on rule layer only
                health_head_output = None

        gate_result = run_gate(labs, mask, gender_str,
                               health_head_output=health_head_output)

        # If gate says patient is healthy — return tiered response, skip Mistral
        if not gate_result['run_full_pipeline']:
            tiered = gate_result['tiered_response']
            structured = _build_tiered_structured_output(patient, tiered)
            narrative  = (
                f"{tiered.get('headline', '')}\n\n"
                f"{tiered.get('detail', '')}\n\n"
                f"Recommendation: {tiered.get('recommendation', '')}\n\n"
                f"DISCLAIMER: This is an AI-generated clinical decision support "
                f"summary. All findings must be reviewed by a qualified clinician."
            )
            timings['gate_s']  = round(time.time() - t_total, 2)
            timings['total_s'] = timings['gate_s']
            return {
                'status':       'ok',
                'error':        None,
                'structured':   structured,
                'narrative':    narrative,
                'c1_raw':       None,
                'verification': {},
                'timings':      timings,
                'patient':      patient,
                'gate_result':  gate_result,
            }

    except Exception as e:
        # Gate failure must NOT block the pipeline — fail open (run full analysis)
        gate_result = {'run_full_pipeline': True, 'error': str(e)}

    timings['gate_s'] = round(time.time() - t_total, 2)

    # ── Component 1 ensemble predict ──────────────────────────────
    try:
        t0 = time.time()
        c1_preds = _ensemble_predict(loaded_models['models'], feats_t)
        timings['component1_s'] = round(time.time() - t0, 2)

    except Exception as e:
        return {
            'status': 'error',
            'error':  f'Component 1 failed: {e}',
            'structured': None, 'narrative': '', 'c1_raw': None,
            'verification': {}, 'timings': timings, 'gate_result': gate_result,
        }

    # ── Mistral verification ───────────────────────────────────────
    verification = {}
    narrative    = ''

    if not skip_mistral:
        try:
            t0 = time.time()
            verification = _call_mistral_verification(patient, c1_preds, ngrok_url)
            timings['verification_s'] = round(time.time() - t0, 2)
        except Exception as e:
            verification = {
                'predictions_correct':          True,
                'corrected_condition':           c1_preds['condition_name'],
                'corrected_risk_level':          c1_preds['risk_level'],
                'corrected_specialist':          SPECIALIST_NAMES.get(
                                                     c1_preds['specialist_code'],
                                                     'General Medicine'),
                'additional_organ_involvement':  [],
                'missed_findings':               [],
                'critical_flags':                [],
                'confidence_note':               f'Mistral verification unavailable: {e}',
            }
            timings['verification_s'] = 0.0

        try:
            t0 = time.time()
            narrative = _call_mistral_narrative(patient, c1_preds, verification, ngrok_url)
            timings['narrative_s'] = round(time.time() - t0, 2)
        except Exception as e:
            narrative = (
                f'[Clinical narrative unavailable — Mistral error: {e}]\n\n'
                f'DISCLAIMER: This is an AI-generated clinical decision support tool. '
                f'All findings must be reviewed by a qualified clinician.'
            )
            timings['narrative_s'] = 0.0

    else:
        verification = {
            'predictions_correct':          True,
            'corrected_condition':           c1_preds['condition_name'],
            'corrected_risk_level':          c1_preds['risk_level'],
            'corrected_specialist':          SPECIALIST_NAMES.get(
                                                 c1_preds['specialist_code'],
                                                 'General Medicine'),
            'additional_organ_involvement':  [],
            'missed_findings':               [],
            'critical_flags':                [],
            'confidence_note':               'Mistral skipped (offline mode)',
        }
        narrative = '[Narrative generation skipped in offline mode]'
        timings['verification_s'] = 0.0
        timings['narrative_s']    = 0.0

    # ── Build structured output ────────────────────────────────────
    structured = _build_structured_output(patient, c1_preds, verification)

    timings['total_s'] = round(time.time() - t_total, 2)

    return {
        'status':       'ok',
        'error':        None,
        'structured':   structured,
        'narrative':    narrative,
        'c1_raw':       c1_preds,
        'verification': verification,
        'timings':      timings,
        'patient':      patient,
        'gate_result':  gate_result,
    }


# ══════════════════════════════════════════════════════════════════
#  CLI QUICK TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python inference.py <path/to/patient.json> [--skip-mistral]')
        sys.exit(1)

    patient_path = sys.argv[1]
    skip_mistral = '--skip-mistral' in sys.argv

    patient = json.loads(Path(patient_path).read_text(encoding='utf-8'))

    print(f'Loading v1 models from {DEFAULT_MODEL_DIR}...')
    loaded = load_models()
    print(f'V1 status: {loaded["status"]} — {loaded["message"]}')

    print(f'Loading v2 models from {DEFAULT_MODEL_DIR_V2}...')
    loaded['v2_models'] = load_models_v2()
    print(f'V2 status: {loaded["v2_models"]["status"]} — {loaded["v2_models"]["message"]}')

    if loaded['status'] != 'ok':
        print('Cannot proceed without v1 models.')
        sys.exit(1)

    print(f'Running inference (skip_mistral={skip_mistral})...')
    result = run_inference(patient, loaded, skip_mistral=skip_mistral)

    if result['status'] != 'ok':
        print(f'ERROR: {result["error"]}')
        sys.exit(1)

    s = result['structured']

    # Show gate decision if available
    gr = result.get('gate_result')
    if gr:
        print(f'\nGATE: run_full_pipeline={gr["run_full_pipeline"]}')
        print(f'      reason: {gr.get("reason", "n/a")}')

    print(f'\n{"="*55}')
    print(f'  KIMS CLINICAL ANALYSIS — {s["age"]}y {s["gender"]}')
    print(f'{"="*55}')

    if s.get('is_tiered_normal'):
        print(f'[TIERED NORMAL RESPONSE]')
        print(f'{s["tiered_headline"]}')
        print(f'{s["tiered_detail"]}')
        print(f'→ {s["tiered_recommendation"]}')
    else:
        print(f'RISK LEVEL:    {s["risk_label"]}')
        print(f'CONDITION:     {s["condition_upper"]}  ({s["condition_confidence"]:.0%})')
        print(f'SPECIALIST:    {s["specialist_name"]}')
        print(f'ORGANS:        {", ".join(s["organs"]) or "None"}')
        print(f'SEVERITY:      {", ".join(s["severity_flags"]) or "None"}')
        print(f'CRITICAL:      {", ".join(s["critical_flags"]) or "None"}')
        print(f'\nNARRATIVE:\n{result["narrative"]}')

    print(f'\nTimings: {result["timings"]}')