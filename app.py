"""
app.py
======
KIMS Clinical AI — Streamlit Frontend

Run with:
    conda activate med_env
    cd D:\\Major_Project\\project\\kims_v3
    streamlit run app.py

Features:
  - Upload PDF or Word lab report (auto-parsed)
  - Manual form fallback if parser misses values
  - Runs Component 1 ensemble + Mistral verification + narrative
  - Styled output cards (risk, condition, specialist, organs, flags)
  - Download as PDF report
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import streamlit as st
import json
import sys
import time
import tempfile
import os
from pathlib import Path

# ── Add project src to path ────────────────────────────────────────
# Adjust this to your project structure
PROJECT_ROOT = Path(r'D:\Major_Project\project\kims_v3')
SRC_C1  = PROJECT_ROOT / 'src' / 'component1'
SRC_C2  = PROJECT_ROOT / 'src' / 'component2'
SRC_PIPELINE = PROJECT_ROOT / 'src' / 'pipeline'

for p in [str(SRC_C1), str(SRC_C2), str(SRC_PIPELINE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title='Name To Be Decided-App (On streamlit Temporarily i dont like ui)',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ══════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
.main .block-container { padding-top: 1.5rem; max-width: 1100px; }

/* ── Risk card ── */
.risk-card {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 1rem;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
.risk-1 { background: #d4edda; color: #155724; border: 2px solid #28a745; }
.risk-2 { background: #fff3cd; color: #856404; border: 2px solid #ffc107; }
.risk-3 { background: #ffe5d0; color: #b94a00; border: 2px solid #fd7e14; }
.risk-4 { background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }

/* ── Info cards ── */
.info-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid #0d6efd;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.info-card h4 {
    margin: 0 0 0.3rem 0;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6c757d;
}
.info-card p {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 600;
    color: #212529;
}

/* ── Organ pill ── */
.organ-pill {
    display: inline-block;
    background: #e7f3ff;
    color: #0d4c9c;
    border: 1px solid #b8d4f5;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.88rem;
    font-weight: 600;
    margin: 3px;
}

/* ── Flag badge ── */
.flag-warning {
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffc107;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.86rem;
    margin: 3px;
}
.flag-critical {
    display: inline-block;
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c2c7;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.86rem;
    margin: 3px;
}

/* ── Confidence badge ── */
.conf-badge {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.86rem;
    font-weight: 600;
    margin-left: 8px;
}
.conf-badge-warn {
    display: inline-block;
    background: #fff8e1;
    color: #f57f17;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.86rem;
    font-weight: 600;
    margin-left: 8px;
}

/* ── Narrative box ── */
.narrative-box {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-size: 0.97rem;
    line-height: 1.7;
    color: #343a40;
}
.narrative-disclaimer {
    font-size: 0.8rem;
    color: #6c757d;
    margin-top: 0.75rem;
    font-style: italic;
}

/* ── Section header ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6c757d;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 2px solid #dee2e6;
}

/* ── Timing strip ── */
.timing-strip {
    font-size: 0.78rem;
    color: #868e96;
    background: #f1f3f5;
    border-radius: 6px;
    padding: 4px 12px;
    margin-top: 0.5rem;
}

/* ── Upload area ── */
.stFileUploader > div { border-radius: 10px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] .stMarkdown h3 { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner='Loading AI models...')
def get_models():
    """Load ensemble models once at startup."""
    try:
        from inference import load_models, load_models_v2
        models = load_models()
        models['v2_models'] = load_models_v2()
        return models
    except ImportError as e:
        return {'status': 'error', 'models': [], 'message': str(e)}


# ══════════════════════════════════════════════════════════════════
#  PDF GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_pdf_report(result: dict, patient: dict) -> bytes:
    """Generate downloadable PDF from inference result using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, black, white, grey
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io

        s   = result['structured']
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
            title='KIMS Clinical AI Report',
        )

        styles = getSampleStyleSheet()
        story  = []

        KIMS_BLUE  = HexColor('#1565C0')
        RISK_PALETTE = {
            1: HexColor('#28a745'),
            2: HexColor('#ffc107'),
            3: HexColor('#fd7e14'),
            4: HexColor('#dc3545'),
        }

        # ── Header ─────────────────────────────────────────────────
        header_style = ParagraphStyle(
            'header', fontSize=18, textColor=white,
            alignment=TA_CENTER, spaceAfter=4, fontName='Helvetica-Bold',
        )
        sub_style = ParagraphStyle(
            'sub', fontSize=10, textColor=HexColor('#CFD8DC'),
            alignment=TA_CENTER, spaceAfter=0, fontName='Helvetica',
        )

        header_table = Table(
            [[Paragraph('KIMS CLINICAL AI REPORT', header_style)],
             [Paragraph('AI-Assisted Clinical Decision Support', sub_style)]],
            colWidths=[17*cm],
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), KIMS_BLUE),
            ('ROUNDEDCORNERS', [8]),
            ('TOPPADDING', (0,0), (-1,-1), 14),
            ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.4*cm))

        # ── Patient Info ────────────────────────────────────────────
        dem = patient.get('demographics', {})
        age = int(dem.get('age', 0))
        gender = 'Male' if dem.get('gender', 1) == 1 else 'Female'

        import datetime
        report_date = datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')

        info_data = [
            ['Patient', f'{age} years, {gender}', 'Report Date', report_date],
        ]
        info_table = Table(info_data, colWidths=[3*cm, 5*cm, 3.5*cm, 5.5*cm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('TEXTCOLOR', (0,0), (-1,-1), HexColor('#424242')),
            ('BACKGROUND', (0,0), (-1,-1), HexColor('#ECEFF1')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('ROUNDEDCORNERS', [6]),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.5*cm))

        # ── Risk Level ──────────────────────────────────────────────
        risk_col  = RISK_PALETTE.get(s['risk_level'], HexColor('#6c757d'))
        risk_data = [[Paragraph(
            f"RISK LEVEL: {s['risk_label']}",
            ParagraphStyle('risk', fontSize=16, textColor=white,
                           fontName='Helvetica-Bold', alignment=TA_CENTER)
        )]]
        risk_table = Table(risk_data, colWidths=[17*cm])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), risk_col),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('ROUNDEDCORNERS', [8]),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.4*cm))

        # ── Key Predictions ─────────────────────────────────────────
        verified_str = '✓ Verified by AI' if s['predictions_verified'] else '⚠ Corrected by AI'
        conf_str     = f"{s['condition_confidence']:.0%}"

        key_data = [
            ['PRIMARY CONDITION',
             f"{s['condition_upper']}  |  {conf_str} confidence  |  {verified_str}"],
            ['RECOMMENDED SPECIALIST', s['specialist_name']],
        ]
        key_table = Table(key_data, colWidths=[5*cm, 12*cm])
        key_table.setStyle(TableStyle([
            ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',    (1,0), (1,-1), 'Helvetica'),
            ('FONTSIZE',    (0,0), (-1,-1), 9),
            ('TEXTCOLOR',   (0,0), (0,-1), HexColor('#1565C0')),
            ('TEXTCOLOR',   (1,0), (1,-1), HexColor('#212529')),
            ('TOPPADDING',  (0,0), (-1,-1), 8),
            ('BOTTOMPADDING',(0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('LINEBELOW',   (0,0), (-1,-2), 0.5, HexColor('#DEE2E6')),
            ('BACKGROUND',  (0,0), (-1,-1), HexColor('#F8F9FA')),
            ('ROUNDEDCORNERS', [6]),
        ]))
        story.append(key_table)
        story.append(Spacer(1, 0.4*cm))

        # ── Organ Involvement ───────────────────────────────────────
        if s['organs']:
            story.append(Paragraph('ORGAN INVOLVEMENT', ParagraphStyle(
                'sh', fontSize=8, textColor=HexColor('#6c757d'),
                fontName='Helvetica-Bold', spaceBefore=4, spaceAfter=4,
            )))
            organ_text = '   '.join(f'■ {o}' for o in s['organs'])
            story.append(Paragraph(organ_text, ParagraphStyle(
                'organs', fontSize=10, textColor=HexColor('#0d4c9c'),
                fontName='Helvetica-Bold', spaceAfter=8,
            )))

        # ── Flags ───────────────────────────────────────────────────
        if s['severity_flags'] or s['critical_flags']:
            flags_data = []
            if s['severity_flags']:
                flags_data.append(['SEVERITY FLAGS',
                                    '   '.join(f'⚠ {f}' for f in s['severity_flags'])])
            if s['critical_flags']:
                flags_data.append(['CRITICAL FLAGS',
                                    '   '.join(f'⚠ {f}' for f in s['critical_flags'])])
            flags_table = Table(flags_data, colWidths=[4*cm, 13*cm])
            flags_table.setStyle(TableStyle([
                ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
                ('FONTSIZE',    (0,0), (-1,-1), 8.5),
                ('TEXTCOLOR',   (0,0), (0,-1), HexColor('#856404')),
                ('TEXTCOLOR',   (1,0), (1,-1), HexColor('#5c2300')),
                ('TOPPADDING',  (0,0), (-1,-1), 6),
                ('BOTTOMPADDING',(0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('BACKGROUND',  (0,0), (-1,-1), HexColor('#FFF8E1')),
                ('LINEBELOW',   (0,0), (-1,-2), 0.5, HexColor('#FFE082')),
                ('ROUNDEDCORNERS', [6]),
            ]))
            story.append(flags_table)
            story.append(Spacer(1, 0.4*cm))

        # ── Narrative ───────────────────────────────────────────────
        story.append(HRFlowable(width='100%', thickness=1,
                                color=HexColor('#DEE2E6'), spaceAfter=8))
        story.append(Paragraph('CLINICAL SUMMARY', ParagraphStyle(
            'sh', fontSize=8, textColor=HexColor('#6c757d'),
            fontName='Helvetica-Bold', spaceAfter=6,
        )))

        narrative_clean = result['narrative'].replace('\n', '<br/>')
        story.append(Paragraph(narrative_clean, ParagraphStyle(
            'narr', fontSize=9.5, leading=16, textColor=HexColor('#343a40'),
            spaceAfter=10,
        )))

        # ── Disclaimer ──────────────────────────────────────────────
        story.append(HRFlowable(width='100%', thickness=0.5,
                                color=HexColor('#DEE2E6'), spaceAfter=6))
        story.append(Paragraph(
            'DISCLAIMER: This is an AI-generated clinical decision support report. '
            'All findings must be reviewed and confirmed by a qualified clinician '
            'before any clinical decision is made. This report does not constitute '
            'a medical diagnosis.',
            ParagraphStyle('disc', fontSize=7.5, textColor=HexColor('#6c757d'),
                           fontName='Helvetica-Oblique', leading=12),
        ))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        st.error(f'PDF generation failed: {e}')
        return b''


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — PATIENT INTAKE FORM
# ══════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar intake form. Returns form values dict."""
    with st.sidebar:
        st.markdown('## 🏥 Patient Intake')
        st.markdown('---')

        st.markdown('#### Demographics')
        age    = st.number_input('Age (years)', min_value=1, max_value=110, value=45, step=1)
        gender = st.selectbox('Gender', ['Female', 'Male'])
        gender_int = 1 if gender == 'Male' else 0

        st.markdown('#### Clinical Info')
        symptoms = st.text_area(
            'Presenting symptoms / history',
            placeholder='e.g. fever for 5 days, jaundice, vomiting...',
            height=80,
        )

        st.markdown('#### Known Comorbidities')
        col1, col2 = st.columns(2)
        with col1:
            dm  = st.checkbox('Diabetes')
            htn = st.checkbox('Hypertension')
            crd = st.checkbox('Cardiac disease')
            rsp = st.checkbox('Respiratory')
        with col2:
            ren = st.checkbox('Renal disease')
            hep = st.checkbox('Hepatic disease')
            uro = st.checkbox('Urological')
            neu = st.checkbox('Neurological')

        st.markdown('#### History')
        col3, col4 = st.columns(2)
        with col3:
            alcohol = st.checkbox('Alcohol use')
            smoking = st.checkbox('Smoking')
        with col4:
            veg  = st.checkbox('Vegetarian diet')
            preg = st.checkbox('Pregnancy')

        st.markdown('#### Vitals *(optional)*')
        spo2  = st.number_input('SpO₂ (%)',       min_value=60, max_value=100, value=98, step=1)
        pulse = st.number_input('Pulse (bpm)',     min_value=30, max_value=200, value=80, step=1)
        rr    = st.number_input('Resp rate (/min)',min_value=8,  max_value=60,  value=16, step=1)

        bp_str = st.text_input('BP (systolic/diastolic)', placeholder='e.g. 120/80')
        bp_s, bp_d = None, None
        if bp_str and '/' in bp_str:
            try:
                parts = bp_str.split('/')
                bp_s = float(parts[0].strip())
                bp_d = float(parts[1].strip())
            except ValueError:
                pass

        st.markdown('---')
        st.markdown(
            '<div style="font-size:0.75rem; color:#6c757d;">'
            '🔒 All data stays on your machine.<br>'
            'Mistral runs locally via Ollama.</div>',
            unsafe_allow_html=True,
        )

    return {
        'age':        age,
        'gender':     gender_int,
        'symptoms':   symptoms,
        'pregnancy':  preg,
        'comorbidities': {
            'diabetes':             int(dm),
            'hypertension':         int(htn),
            'cardiac_disease':      int(crd),
            'respiratory_disease':  int(rsp),
            'renal_disease':        int(ren),
            'hepatic_disease':      int(hep),
            'urological_condition': int(uro),
            'neurological_condition': int(neu),
        },
        'history': {
            'known_diabetes':   int(dm),
            'alcohol':          int(alcohol),
            'smoking':          int(smoking),
            'diet_vegetarian':  int(veg),
            'sleep_adequate':   1,
        },
        'vitals': {
            'spo2':              float(spo2),
            'pulse':             float(pulse),
            'respiratory_rate':  float(rr),
            'bp_systolic':       bp_s,
            'bp_diastolic':      bp_d,
            'temperature_afebrile': 1,
        },
    }


# ══════════════════════════════════════════════════════════════════
#  OUTPUT RENDERING
# ══════════════════════════════════════════════════════════════════

def render_results(result: dict, patient: dict):
    """Render the full styled inference output."""
    if result['status'] != 'ok':
        st.error(f"❌ Inference failed: {result['error']}")
        return

    s = result['structured']

    # ── Risk level card ────────────────────────────────────────────
    st.markdown(
        f'<div class="risk-card risk-{s["risk_level"]}">'
        f'⚡ RISK LEVEL: {s["risk_label"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Two-column main cards ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        conf_pct = f"{s['condition_confidence']:.0%}"
        conf_cls = 'conf-badge' if s['condition_confidence'] >= 0.70 else 'conf-badge-warn'
        verified = '✅ Verified' if s['predictions_verified'] else '⚠️ Corrected'
        st.markdown(
            f'<div class="info-card">'
            f'<h4>Primary Condition</h4>'
            f'<p>{s["condition_upper"]}'
            f'<span class="{conf_cls}">{conf_pct}</span>'
            f'<span style="font-size:0.8rem;color:#6c757d;margin-left:6px;">{verified}</span>'
            f'</p></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="info-card" style="border-left-color:#198754;">'
            f'<h4>Recommended Specialist</h4>'
            f'<p>{s["specialist_name"]}</p></div>',
            unsafe_allow_html=True,
        )

    # ── Patient info strip ─────────────────────────────────────────
    dem = patient.get('demographics', {})
    age = int(dem.get('age', 0))
    gender = 'Male' if dem.get('gender', 1) == 1 else 'Female'
    st.markdown(
        f'<div style="font-size:0.85rem;color:#6c757d;margin-bottom:0.8rem;">'
        f'Patient: {age}y {gender}</div>',
        unsafe_allow_html=True,
    )

    # ── Organ Involvement ──────────────────────────────────────────
    st.markdown('<div class="section-header">Organ Involvement</div>',
                unsafe_allow_html=True)
    if s['organs']:
        pills = ''.join(f'<span class="organ-pill">✓ {o}</span>' for o in s['organs'])
        st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#6c757d;font-size:0.9rem;">None identified</span>',
                    unsafe_allow_html=True)

    st.markdown('')

    # ── Severity + Critical flags ──────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Severity Flags</div>',
                    unsafe_allow_html=True)
        if s['severity_flags']:
            badges = ''.join(
                f'<span class="flag-warning">⚠ {f}</span>'
                for f in s['severity_flags']
            )
            st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#6c757d;font-size:0.9rem;">None</span>',
                        unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-header">Critical Flags</div>',
                    unsafe_allow_html=True)
        if s['critical_flags']:
            badges = ''.join(
                f'<span class="flag-critical">🚨 {f}</span>'
                for f in s['critical_flags']
            )
            st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#6c757d;font-size:0.9rem;">None</span>',
                        unsafe_allow_html=True)

    st.markdown('')

    # ── Missed findings ────────────────────────────────────────────
    if s.get('missed_findings'):
        with st.expander('🔍 Additional Findings (from AI review)', expanded=False):
            for f in s['missed_findings']:
                st.markdown(f'→ {f}')

    # ── Clinical narrative ─────────────────────────────────────────
    st.markdown('<div class="section-header">Clinical Summary</div>',
                unsafe_allow_html=True)

    # Split narrative from disclaimer
    narrative_text = result['narrative']
    if 'DISCLAIMER' in narrative_text:
        parts = narrative_text.split('DISCLAIMER')
        main_text    = parts[0].strip()
        disc_text    = 'DISCLAIMER' + parts[1].strip()
    else:
        main_text = narrative_text
        disc_text = ('DISCLAIMER: This is an AI-generated clinical decision support summary. '
                     'All findings must be reviewed and confirmed by a qualified clinician '
                     'before any clinical decision is made.')

    st.markdown(
        f'<div class="narrative-box">{main_text}'
        f'<div class="narrative-disclaimer">{disc_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('')

    # ── Timing strip ──────────────────────────────────────────────
    t = result['timings']
    st.markdown(
        f'<div class="timing-strip">'
        f'⏱ Component 1: {t.get("component1_s", 0):.2f}s  |  '
        f'Verification: {t.get("verification_s", 0):.1f}s  |  '
        f'Narrative: {t.get("narrative_s", 0):.1f}s  |  '
        f'Total: {t.get("total_s", 0):.1f}s'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── PDF download ────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    pdf_bytes = generate_pdf_report(result, patient)
    if pdf_bytes:
        st.download_button(
            label='📄 Download Report as PDF',
            data=pdf_bytes,
            file_name=f'KIMS_report_{age}y_{gender}_{int(time.time())}.pdf',
            mime='application/pdf',
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
#  MANUAL LAB OVERRIDE EXPANDER
# ══════════════════════════════════════════════════════════════════

def render_lab_override(parsed_labs: dict) -> dict:
    """
    Expander for manually reviewing / overriding parsed lab values.
    Returns the final (possibly edited) lab values dict.
    """
    with st.expander('🔬 Review / Edit Extracted Lab Values', expanded=False):
        st.markdown(
            '_Parser found these values. Edit any incorrect ones, '
            'or fill in missing ones._'
        )

        FIELD_LABELS = {
            'hemoglobin':      'Hemoglobin (g/dL)',
            'platelets':       'Platelets (/cumm)',
            'wbc':             'WBC (/cumm)',
            'esr':             'ESR (mm/hr)',
            'bilirubin_total': 'Total Bilirubin (mg/dL)',
            'albumin':         'Albumin (g/dL)',
            'ferritin':        'Ferritin (ng/mL)',
            'serum_iron':      'Serum Iron (µg/dL)',
            'creatinine':      'Creatinine (mg/dL)',
            'urea':            'Urea (mg/dL)',
            'grbs':            'Blood Glucose (mg/dL)',
            'hba1c':           'HbA1c (%)',
            'sodium':          'Sodium (mmol/L)',
            'potassium':       'Potassium (mmol/L)',
            'chloride':        'Chloride (mmol/L)',
            'sgot':            'SGOT/AST (U/L)',
            'sgpt':            'SGPT/ALT (U/L)',
            'alp':             'ALP (U/L)',
            'ggt':             'GGT (U/L)',
            'crp':             'CRP (mg/L)',
            'uric_acid':       'Uric Acid (mg/dL)',
            'pcv':             'PCV/Haematocrit (%)',
            'mcv':             'MCV (fl)',
            'rdw':             'RDW-CV (%)',
            'neutrophils_pct': 'Neutrophils (%)',
            'lymphocytes_pct': 'Lymphocytes (%)',
        }

        cols = st.columns(3)
        edited = {}
        for idx, (field, label) in enumerate(FIELD_LABELS.items()):
            col = cols[idx % 3]
            current = parsed_labs.get(field)
            val = col.number_input(
                label,
                value=float(current) if current is not None else 0.0,
                min_value=0.0,
                step=0.1,
                format='%.2f',
                key=f'lab_{field}',
            )
            if val > 0.0:
                edited[field] = val

        return edited


# ══════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar intake form ────────────────────────────────────────
    form_data = render_sidebar()

    # ── Header ─────────────────────────────────────────────────────
    st.markdown('# 🏥 Name To Be Decided-App (On streamlit Temporarily i dont like ui)')
    st.markdown(
        '<span style="color:#6c757d;font-size:0.95rem;">'
        'AI-powered clinical decision support — upload a lab report or fill '
        'in values manually</span>',
        unsafe_allow_html=True,
    )
    st.markdown('---')

    # ── Model status ───────────────────────────────────────────────
    loaded_models = get_models()
    if loaded_models['status'] != 'ok':
        st.error(
            f'⚠️ Models failed to load: {loaded_models.get("message", "unknown")}. '
            f'Ensure `kims_v3/models/component1/ensemble_config.json` exists '
            f'and the med_env is active.'
        )
        st.stop()
    else:
        v2_status = loaded_models.get('v2_models', {}).get('status', 'not loaded')
        st.success(f'✅ {loaded_models["message"]} | Gate (v2): {v2_status}', icon=None)
        st.markdown('')

    # ── File upload ─────────────────────────────────────────────────
    st.markdown('### 📂 Upload Lab Report')
    uploaded_file = st.file_uploader(
        'Upload a PDF or Word document lab report',
        type=['pdf', 'docx'],
        help='Supports Kanva PDF, KIMS Srishti HTML (zip/folder), and Word formats.',
    )

    parsed_labs  = {}
    parse_status = ''
    source_type  = ''

    if uploaded_file is not None:
        # Save to temp file
        suffix = '.pdf' if uploaded_file.name.endswith('.pdf') else '.docx'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner('Parsing lab report...'):
                from pdf_parser import parse_lab_report
                parse_result = parse_lab_report(tmp_path)

            parsed_labs = parse_result.get('lab_values', {})
            source_type = parse_result.get('source_type', 'unknown')
            n_found     = parse_result.get('n_labs_found', 0)
            warnings    = parse_result.get('parse_warnings', [])

            # Auto-fill demographics from parsed report if not already set
            parsed_dem = parse_result.get('demographics', {})

            if n_found > 0:
                parse_status = f'✅ Parsed **{n_found} lab values** from {source_type.replace("_", " ").title()}'
            else:
                parse_status = f'⚠️ Parser found 0 values from {source_type}. Please fill in manually below.'

            if warnings:
                for w in warnings:
                    st.warning(f'Parser: {w}')

        except Exception as e:
            parse_status = f'❌ Parse error: {e}'
        finally:
            os.unlink(tmp_path)

        if parse_status:
            if parse_status.startswith('✅'):
                st.success(parse_status)
            elif parse_status.startswith('⚠️'):
                st.warning(parse_status)
            else:
                st.error(parse_status)

    # ── Lab value review/override ──────────────────────────────────
    if uploaded_file is not None or st.checkbox('Enter lab values manually', value=False):
        final_labs = render_lab_override(parsed_labs)
    else:
        final_labs = parsed_labs

    st.markdown('')

    # ── Run Inference button ───────────────────────────────────────
    st.markdown('---')
    col_run, col_offline = st.columns([3, 1])

    with col_run:
        run_btn = st.button(
            '🚀 Run Clinical Analysis',
            use_container_width=True,
            type='primary',
            disabled=(not final_labs),
        )
    with col_offline:
        skip_mistral = st.checkbox(
            'Offline mode\n(skip Mistral)',
            help='Skip Mistral — use Component 1 predictions only. Much faster.',
        )

    if not final_labs:
        st.info('Upload a lab report or enter values manually to run analysis.')

    # ── Run inference ──────────────────────────────────────────────
    if run_btn and final_labs:
        with st.spinner('Running AI analysis... (~20-30 seconds with Mistral)'):
            try:
                from rule_scorer import build_patient_json
                from inference import run_inference

                patient_json = build_patient_json(
                    lab_values=final_labs,
                    age=form_data['age'],
                    gender=form_data['gender'],
                    symptoms_text=form_data['symptoms'],
                    comorbidities=form_data['comorbidities'],
                    vitals=form_data['vitals'],
                    history=form_data['history'],
                )

                result = run_inference(
                    patient=patient_json,
                    loaded_models=loaded_models,
                    skip_mistral=skip_mistral,
                )

                # Store in session state to persist across reruns
                st.session_state['last_result']  = result
                st.session_state['last_patient'] = patient_json

            except Exception as e:
                st.error(f'Pipeline error: {e}')
                import traceback
                st.code(traceback.format_exc())

    # ── Render stored result ───────────────────────────────────────
    if 'last_result' in st.session_state:
        st.markdown('---')
        st.markdown('### 📋 Analysis Results')
        render_results(
            st.session_state['last_result'],
            st.session_state['last_patient'],
        )

    # ── Footer ─────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown(
        '<div style="font-size:0.75rem;color:#adb5bd;text-align:center;">'
        'KIMS Clinical AI Pipeline — For research and decision support only. '
        'Not a substitute for clinical judgement.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
