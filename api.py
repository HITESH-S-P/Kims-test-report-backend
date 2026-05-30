r"""
api.py
======
FastAPI backend for the KIMS Clinical AI.

Wraps the existing pipeline (inference.py + pdf_parser.py + rule_scorer.py)
as a REST API so the React Native frontend can talk to it over HTTP.
Nothing about the AI logic changes — this is a thin transport layer.

Place this file at:   D:\Major_Project\project\kims_v3\api.py   (next to app.py)

Run with:
    conda activate med
    cd D:\Major_Project\project\kims_v3
    python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Interactive docs (auto-generated) at:  http://localhost:8000/docs

Endpoints:
    GET  /                     → API info
    GET  /health               → models loaded? Ollama reachable?
    GET  /models/status        → v1 / v2 load detail
    POST /parse                → upload PDF/DOCX → extracted labs + demographics
    POST /analyze              → final labs + form data → full clinical result
    GET  /report/pdf/{id}      → download PDF for a prior /analyze result
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # Windows OpenMP fix (same as app.py)

import sys
import json
import time
import uuid
import tempfile
import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ── Make project modules importable ────────────────────────────────
PROJECT_ROOT = Path(r'D:\Major_Project\project\kims_v3')
SRC_C1       = PROJECT_ROOT / 'src' / 'component1'
SRC_C2       = PROJECT_ROOT / 'src' / 'component2'
SRC_PIPELINE = PROJECT_ROOT / 'src' / 'pipeline'
SRC_PV2      = PROJECT_ROOT / 'src' / 'pipeline_v2' / 'Lab_ml_train_2'

for p in [SRC_C1, SRC_C2, SRC_PIPELINE, SRC_PV2]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# These imports work once the paths above are set
from inference import load_models, load_models_v2, run_inference   # noqa: E402
from pdf_parser import parse_lab_report                            # noqa: E402
from rule_scorer import build_patient_json                         # noqa: E402

OLLAMA_URL = 'http://localhost:11434'

# In-memory cache of recent analysis results, keyed by report_id.
# Lets the frontend request a PDF by id instead of POSTing the whole
# result back. Fine for a local single-user app; swap for Redis if scaled.
_RESULT_CACHE: Dict[str, dict] = {}
_CACHE_MAX = 50   # keep the last N results


# ══════════════════════════════════════════════════════════════════
#  LIFESPAN — load models once at startup (replaces st.cache_resource)
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('[startup] Loading v1 ensemble models...')
    models = load_models()
    print(f'[startup] v1: {models["status"]} — {models.get("message", "")}')

    print('[startup] Loading v2 (health-head) models...')
    models['v2_models'] = load_models_v2()
    print(f'[startup] v2: {models["v2_models"]["status"]} — '
          f'{models["v2_models"].get("message", "")}')

    app.state.models = models
    yield
    # nothing to tear down — torch frees on process exit
    print('[shutdown] done.')


app = FastAPI(
    title='KIMS Clinical AI API',
    description='REST backend for the KIMS clinical decision support pipeline.',
    version='1.0.0',
    lifespan=lifespan,
)

# CORS — permissive for local dev. Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# ══════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════

class Comorbidities(BaseModel):
    diabetes: int = 0
    hypertension: int = 0
    cardiac_disease: int = 0
    respiratory_disease: int = 0
    renal_disease: int = 0
    hepatic_disease: int = 0
    urological_condition: int = 0
    neurological_condition: int = 0


class Vitals(BaseModel):
    spo2: float = 98.0
    pulse: float = 80.0
    respiratory_rate: float = 16.0
    bp_systolic: Optional[float] = None
    bp_diastolic: Optional[float] = None
    temperature_afebrile: int = 1


class History(BaseModel):
    known_diabetes: int = 0
    alcohol: int = 0
    smoking: int = 0
    diet_vegetarian: int = 0
    sleep_adequate: int = 1


class AnalyzeRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: int = Field(..., description='1 = Male, 0 = Female')
    symptoms: str = ''
    lab_values: Dict[str, float] = Field(
        ..., description='Final (reviewed/edited) lab values'
    )
    comorbidities: Comorbidities = Comorbidities()
    vitals: Vitals = Vitals()
    history: History = History()
    pregnancy: int = 0
    skip_mistral: bool = False


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════

def _ollama_up() -> bool:
    """Lightweight check that Ollama is serving locally."""
    try:
        r = httpx.get(f'{OLLAMA_URL}/api/tags', timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _cache_result(result: dict, patient: dict) -> str:
    """Store a result for later PDF retrieval. Returns a report_id."""
    rid = uuid.uuid4().hex[:12]
    _RESULT_CACHE[rid] = {'result': result, 'patient': patient}
    # Trim oldest entries if cache grows too large
    if len(_RESULT_CACHE) > _CACHE_MAX:
        oldest = next(iter(_RESULT_CACHE))
        _RESULT_CACHE.pop(oldest, None)
    return rid


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get('/')
def root():
    return {
        'service': 'KIMS Clinical AI API',
        'version': '1.0.0',
        'docs': '/docs',
        'endpoints': ['/health', '/models/status', '/parse', '/analyze',
                      '/report/pdf/{report_id}'],
    }


@app.get('/health')
def health():
    models = app.state.models
    v1_ok = models.get('status') == 'ok'
    v2_ok = models.get('v2_models', {}).get('status') == 'ok'
    ollama_ok = _ollama_up()
    return {
        'status': 'ok' if (v1_ok and ollama_ok) else 'degraded',
        'v1_models_loaded': v1_ok,
        'v2_gate_loaded': v2_ok,
        'ollama_reachable': ollama_ok,
        'note': ('Gate (v2) disabled — running v1 only'
                 if (v1_ok and not v2_ok) else ''),
    }


@app.get('/models/status')
def models_status():
    models = app.state.models
    v2 = models.get('v2_models', {})
    return {
        'v1': {
            'status': models.get('status'),
            'message': models.get('message'),
            'n_models': len(models.get('models', [])),
        },
        'v2': {
            'status': v2.get('status'),
            'message': v2.get('message'),
            'n_models': len(v2.get('models', [])),
        },
    }


@app.post('/parse')
async def parse(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX lab report. Returns extracted lab values and
    any demographics found in the report header, for the frontend to
    show in a review/override screen before /analyze.
    """
    name = (file.filename or '').lower()
    if not (name.endswith('.pdf') or name.endswith('.docx')):
        raise HTTPException(400, 'Only .pdf or .docx files are supported.')

    suffix = '.pdf' if name.endswith('.pdf') else '.docx'
    tmp_path = None
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        parsed = parse_lab_report(tmp_path)
        return {
            'status': 'ok',
            'lab_values':   parsed.get('lab_values', {}),
            'demographics': parsed.get('demographics', {}),
            'source_type':  parsed.get('source_type', 'unknown'),
            'n_labs_found': parsed.get('n_labs_found', 0),
            'warnings':     parsed.get('parse_warnings', []),
        }
    except Exception as e:
        raise HTTPException(500, f'Parse failed: {e}')
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post('/analyze')
def analyze(req: AnalyzeRequest):
    """
    Run the full clinical pipeline on a set of final lab values + form data.
    The gate decides whether Mistral runs; tiered-normal results skip it.
    Returns the structured result plus a report_id for PDF retrieval.
    """
    models = app.state.models
    if models.get('status') != 'ok':
        raise HTTPException(503, f'Models not loaded: {models.get("message")}')

    if not req.lab_values:
        raise HTTPException(400, 'No lab values provided.')

    try:
        patient = build_patient_json(
            lab_values=req.lab_values,
            age=req.age,
            gender=req.gender,
            symptoms_text=req.symptoms,
            comorbidities=req.comorbidities.model_dump(),
            vitals=req.vitals.model_dump(),
            history=req.history.model_dump(),
        )
    except Exception as e:
        raise HTTPException(500, f'Failed to build patient JSON: {e}')

    result = run_inference(
        patient=patient,
        loaded_models=models,
        skip_mistral=req.skip_mistral,
    )

    if result['status'] != 'ok':
        raise HTTPException(500, f'Inference failed: {result.get("error")}')

    report_id = _cache_result(result, patient)

    gate = result.get('gate_result') or {}
    return {
        'status':       'ok',
        'report_id':    report_id,
        'structured':   result['structured'],
        'narrative':    result['narrative'],
        'gate': {
            'ran_full_pipeline': gate.get('run_full_pipeline', True),
            'reason':            gate.get('reason', ''),
        },
        'timings':      result['timings'],
    }


@app.get('/report/pdf/{report_id}')
def report_pdf(report_id: str):
    """Generate and return the PDF for a cached /analyze result."""
    entry = _RESULT_CACHE.get(report_id)
    if not entry:
        raise HTTPException(404, 'Report not found or expired.')

    pdf_bytes = _generate_pdf(entry['result'], entry['patient'])
    if not pdf_bytes:
        raise HTTPException(500, 'PDF generation failed.')

    s = entry['result']['structured']
    fname = f'KIMS_report_{s["age"]}y_{s["gender"]}_{int(time.time())}.pdf'
    return Response(
        content=pdf_bytes,
        media_type='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{fname}"'},
    )


# ══════════════════════════════════════════════════════════════════
#  PDF GENERATION  (reportlab — ported from app.py, robust to tiered)
# ══════════════════════════════════════════════════════════════════

def _generate_pdf(result: dict, patient: dict) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER
        import io

        s   = result['structured']
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
            title='KIMS Clinical AI Report',
        )
        getSampleStyleSheet()
        story = []

        KIMS_BLUE = HexColor('#1565C0')
        RISK_PALETTE = {1: HexColor('#28a745'), 2: HexColor('#ffc107'),
                        3: HexColor('#fd7e14'), 4: HexColor('#dc3545')}

        header_style = ParagraphStyle('header', fontSize=18, textColor=white,
                                      alignment=TA_CENTER, spaceAfter=4,
                                      fontName='Helvetica-Bold')
        sub_style = ParagraphStyle('sub', fontSize=10, textColor=HexColor('#CFD8DC'),
                                   alignment=TA_CENTER, fontName='Helvetica')

        header_table = Table(
            [[Paragraph('KIMS CLINICAL AI REPORT', header_style)],
             [Paragraph('AI-Assisted Clinical Decision Support', sub_style)]],
            colWidths=[17*cm])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), KIMS_BLUE),
            ('ROUNDEDCORNERS', [8]),
            ('TOPPADDING', (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.4*cm))

        report_date = datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')
        info_table = Table(
            [['Patient', f'{s["age"]} years, {s["gender"]}',
              'Report Date', report_date]],
            colWidths=[3*cm, 5*cm, 3.5*cm, 5.5*cm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#424242')),
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#ECEFF1')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('ROUNDEDCORNERS', [6]),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.5*cm))

        # Risk banner
        risk_col = RISK_PALETTE.get(s['risk_level'], HexColor('#6c757d'))
        risk_table = Table([[Paragraph(
            f"RISK LEVEL: {s['risk_label']}",
            ParagraphStyle('risk', fontSize=16, textColor=white,
                           fontName='Helvetica-Bold', alignment=TA_CENTER))]],
            colWidths=[17*cm])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), risk_col),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('ROUNDEDCORNERS', [8]),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.4*cm))

        # Key predictions
        verified_str = ('Verified by AI' if s['predictions_verified']
                        else 'Corrected by AI')
        conf_str = f"{s['condition_confidence']:.0%}"
        key_table = Table([
            ['PRIMARY CONDITION',
             f"{s['condition_upper']}  |  {conf_str} confidence  |  {verified_str}"],
            ['RECOMMENDED SPECIALIST', s['specialist_name']],
        ], colWidths=[5*cm, 12*cm])
        key_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#1565C0')),
            ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#212529')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, HexColor('#DEE2E6')),
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#F8F9FA')),
            ('ROUNDEDCORNERS', [6]),
        ]))
        story.append(key_table)
        story.append(Spacer(1, 0.4*cm))

        # Organs
        if s.get('organs'):
            story.append(Paragraph('ORGAN INVOLVEMENT', ParagraphStyle(
                'sh', fontSize=8, textColor=HexColor('#6c757d'),
                fontName='Helvetica-Bold', spaceBefore=4, spaceAfter=4)))
            organ_text = '   '.join(f'■ {o}' for o in s['organs'])
            story.append(Paragraph(organ_text, ParagraphStyle(
                'organs', fontSize=10, textColor=HexColor('#0d4c9c'),
                fontName='Helvetica-Bold', spaceAfter=8)))

        # Flags
        if s.get('severity_flags') or s.get('critical_flags'):
            flags_data = []
            if s.get('severity_flags'):
                flags_data.append(['SEVERITY FLAGS',
                                   '   '.join(f'! {f}' for f in s['severity_flags'])])
            if s.get('critical_flags'):
                flags_data.append(['CRITICAL FLAGS',
                                   '   '.join(f'! {f}' for f in s['critical_flags'])])
            flags_table = Table(flags_data, colWidths=[4*cm, 13*cm])
            flags_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#856404')),
                ('TEXTCOLOR', (1, 0), (1, -1), HexColor('#5c2300')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 0), (-1, -1), HexColor('#FFF8E1')),
                ('LINEBELOW', (0, 0), (-1, -2), 0.5, HexColor('#FFE082')),
                ('ROUNDEDCORNERS', [6]),
            ]))
            story.append(flags_table)
            story.append(Spacer(1, 0.4*cm))

        # Narrative
        story.append(HRFlowable(width='100%', thickness=1,
                                color=HexColor('#DEE2E6'), spaceAfter=8))
        story.append(Paragraph('CLINICAL SUMMARY', ParagraphStyle(
            'sh', fontSize=8, textColor=HexColor('#6c757d'),
            fontName='Helvetica-Bold', spaceAfter=6)))
        narrative_clean = (result.get('narrative') or '').replace('\n', '<br/>')
        story.append(Paragraph(narrative_clean, ParagraphStyle(
            'narr', fontSize=9.5, leading=16, textColor=HexColor('#343a40'),
            spaceAfter=10)))

        story.append(HRFlowable(width='100%', thickness=0.5,
                                color=HexColor('#DEE2E6'), spaceAfter=6))
        story.append(Paragraph(
            'DISCLAIMER: This is an AI-generated clinical decision support report. '
            'All findings must be reviewed and confirmed by a qualified clinician '
            'before any clinical decision is made. This report does not constitute '
            'a medical diagnosis.',
            ParagraphStyle('disc', fontSize=7.5, textColor=HexColor('#6c757d'),
                           fontName='Helvetica-Oblique', leading=12)))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        print(f'[pdf] generation failed: {e}')
        return b''


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=True)