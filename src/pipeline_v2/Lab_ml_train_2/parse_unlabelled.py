"""
parse_unlabelled_v2.py
======================
Complete rewrite of the KIMS Srishti HTML parser.

FIXES FROM V1:
  1. All dates preserved per-field (no silent first-wins)
     → admission value (earliest date) stored as primary
     → all dated values stored in lab_timeline for audit
  2. MCH and MCHC now extracted (were in HTML, missing from schema)
  3. Direct bilirubin, total protein now extracted
  4. Lipid profile extracted (triglycerides, HDL, cholesterol etc.)
  5. Urine analysis extracted as a separate structured section
     (qualitative fields handled as categorical integers, not floats)
  6. Demographics extracted from header (age, gender, patient type)
  7. HbA1c range parsed correctly (multi-line interpretation text)
  8. Duplicate reports (same date, same content) deduplicated silently
  9. Peripheral smear impression extracted as text field
  10. Parser logs which date each field came from for full auditability

URINE ANALYSIS DESIGN NOTE:
  Urine fields are qualitative (NIL/ABSENT/NEGATIVE/NORMAL = 0,
  anything else = 1, numeric values stored directly).
  These go into a separate 'urine_features' dict, NOT lab_features,
  because they are not continuous-valued and cannot be normalised
  the same way as serum labs. The training pipeline handles them
  separately. They are still useful for the normality gate.
"""

import re, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("pip install beautifulsoup4")


# ═══════════════════════════════════════════════════════════════════
#  SCHEMA MAPS
# ═══════════════════════════════════════════════════════════════════

# Primary lab fields (continuous numeric → float)
LAB_TEST_MAP = {
    # Haematology
    'HEMOGLOBIN': 'hemoglobin',
    'HAEMOGLOBIN': 'hemoglobin',
    'HB': 'hemoglobin',
    'HEMOGLOBIN (SLS HEMOGLOBIN(AUTOMATED))': 'hemoglobin',
    'HEMOGLOBIN (SLS)': 'hemoglobin',
    'PLATELET COUNT': 'platelets',
    'PLATELET COUNT (AUTOMATED/MANUAL)': 'platelets',
    'PLATELETS': 'platelets',
    'TOTAL COUNT': 'wbc',
    'TOTAL LEUCOCYTE COUNT': 'wbc',
    'TOTAL LEUCOCYTE COUNT (FLOW CYTOMETRY)': 'wbc',
    'TOTAL COUNT(TC)': 'wbc',
    'TC': 'wbc',
    'WBC': 'wbc',
    'WBC COUNT': 'wbc',
    'ESR': 'esr',
    'ESR (AUTOMATED)': 'esr',
    'ESR (WESTERGREN METHOD)': 'esr',
    'NEUTROPHILS': 'neutrophils_pct',
    'NEUTROPHILS (FLOW CYTOMETRY)': 'neutrophils_pct',
    'LYMPHOCYTES': 'lymphocytes_pct',
    'LYMPHOCYTES (FLOW CYTOMETRY)': 'lymphocytes_pct',
    'MONOCYTES': 'monocytes_pct',
    'MONOCYTES (FLOW CYTOMETRY)': 'monocytes_pct',
    'EOSINOPHILS': 'eosinophils_pct',
    'EOSINOPHILS (FLOW CYTOMETRY)': 'eosinophils_pct',
    'BASOPHILS': 'basophils_pct',
    'BASOPHILS (FLOW CYTOMETRY)': 'basophils_pct',
    'RBC COUNT': 'rbc_count',
    'RBC COUNT (AUTOMATED)': 'rbc_count',
    'PACKED CELL VOLUME': 'pcv',
    'PACKED CELL VOLUME (CALCULATED)': 'pcv',
    'PACKED CELL VOLUME(PCV)': 'pcv',
    'MCV': 'mcv',
    'MCV (CALCULATED)': 'mcv',
    'MCH': 'mch',
    'MCH (CALCULATED)': 'mch',
    'MCHC': 'mchc',
    'MCHC (CALCULATED)': 'mchc',
    'RDW-CV': 'rdw',
    'RDW-CV (AUTOMATED)': 'rdw',

    # Biochemistry — liver
    'TOTAL BILIRUBIN': 'bilirubin_total',
    'TOTAL BILIRUBIN(METHOD-JENDRASSIK - GROF)': 'bilirubin_total',
    'DIRECT BILIRUBIN': 'direct_bilirubin',
    'INDIRECT BILIRUBIN': 'indirect_bilirubin',
    'SERUM ALBUMIN': 'albumin',
    'SERUM ALBUMIN(METHOD-BCG)': 'albumin',
    'ALBUMIN': 'albumin',
    'TOTAL PROTEIN': 'total_protein',
    'SGOT': 'sgot',
    'SGOT(AST)': 'sgot',
    'SGOT(AST)(METHOD-IFCC WITH P5P)': 'sgot',
    'S.G.O.T (A.S.T)': 'sgot',
    'SGPT': 'sgpt',
    'SGPT(ALT)': 'sgpt',
    'SGPT(ALT)(METHOD-IFCC WITH P5P)': 'sgpt',
    'S.G.P.T (A.L.T)': 'sgpt',
    'ALKALINE PHOSPHATASE': 'alp',
    'ALKALINE PHOSPHATASE(METHOD-IFCC)': 'alp',
    'GGT': 'ggt',
    'GAMMA GLUTAMYL TRASNSFERASE (GGT)': 'ggt',
    'GAMMA GLUTAMYL TRASNSFERASE (GGT)(METHOD-IFCC)': 'ggt',

    # Biochemistry — renal
    'CREATININE': 'creatinine',
    'SERUM CREATININE': 'creatinine',
    'CREATININE (METHOD-JAFFES IDMS TRACEABLE)': 'creatinine',
    'CREATININE(METHOD-JAFFES IDMS TRACEABLE)': 'creatinine',
    'UREA': 'urea',
    'BLOOD UREA': 'urea',
    'UREA (METHOD-GLDH/UREASE)': 'urea',
    'UREA(METHOD-GLDH/UREASE)': 'urea',
    'URIC ACID': 'uric_acid',
    'SERUM URIC ACID': 'uric_acid',
    'SERUM URIC ACID(METHOD-URICASE)': 'uric_acid',

    # Biochemistry — electrolytes
    'SODIUM': 'sodium',
    'SODIUM(METHOD-ISE INDIRECT)': 'sodium',
    'POTASSIUM': 'potassium',
    'POTASSIUM(METHOD-ISE INDIRECT)': 'potassium',
    'CHLORIDE': 'chloride',
    'CHLORIDE(METHOD-ISE INDIRECT)': 'chloride',

    # Biochemistry — metabolic
    'GRBS': 'grbs',
    'GLUCOSE FASTING': 'grbs',
    'BLOOD GLUCOSE FASTING': 'grbs',
    'GLUCOSE RANDOM': 'grbs',
    'RANDOM BLOOD SUGAR': 'grbs',
    'FBS': 'grbs',
    'RBS': 'grbs',
    'GLYCOSYLATED HB (HBA1C)': 'hba1c',
    'GLYCOSYLATED HAEMOGLOBIN(HBA1C)': 'hba1c',
    'HBA1C': 'hba1c',
    'GLYCOSYLATED HEMOGLOBIN': 'hba1c',
    'BMI': 'bmi',
    'BODY MASS INDEX': 'bmi',

    # Iron studies
    'FERRITIN': 'ferritin',
    'SERUM FERRITIN': 'ferritin',
    'IRON': 'serum_iron',
    'SERUM IRON': 'serum_iron',
    'SERUM IRON(METHOD-FERROZINE)': 'serum_iron',
    'TRANSFERRIN SATURATION': 'tsat',
    'TSAT': 'tsat',
    'TRANSFERRIN SATURATION(METHOD-CALCULATED)': 'tsat',

    # Cardiac
    'EF': 'ef_percent',
    'EJECTION FRACTION': 'ef_percent',

    # Inflammatory
    'CRP': 'crp',
    'C-REACTIVE PROTEIN': 'crp',

    # Lipid profile
    'T.CHOLESTEROL': 'cholesterol_total',
    'TRIGLYCERIDES': 'triglycerides',
    'HDL': 'hdl',
    'LDL': 'ldl',
    'VLDL': 'vldl',

    # Thyroid
    'T3': 'thyroid_t3',
    'T4': 'thyroid_t4',
    'TSH': 'thyroid_tsh',
}

# Urine analysis fields — qualitative values mapped to integer
# 0 = normal/negative/nil/absent, 1 = abnormal/present, numeric = stored as-is
URINE_TEST_MAP = {
    'SPECIFIC GRAVITY': 'urine_specific_gravity',
    'SPECIFIC GRAVITY(AUTOMATED)': 'urine_specific_gravity',
    'PH': 'urine_ph',
    'PROTEIN': 'urine_protein',
    'GLUCOSE': 'urine_glucose',
    'KETONE BODIES': 'urine_ketones',
    'UROBILINOGEN': 'urine_urobilinogen',
    'BILIRUBIN': 'urine_bilirubin',
    'BLOOD': 'urine_blood',
    'NITRITE': 'urine_nitrite',
    'LEUCOCYTES': 'urine_leucocytes',
    'PUS CELLS': 'urine_pus_cells',
    'EPITHELIAL CELLS': 'urine_epithelial_cells',
    'RBCS': 'urine_rbc',
    'CASTS': 'urine_casts',
    'CRYSTALS': 'urine_crystals',
    'BACTERIA': 'urine_bacteria',
}

# These qualitative values all mean "negative / normal / absent"
NEGATIVE_QUALITATIVE = {
    'NIL', 'ABSENT', 'NEGATIVE', 'NORMAL', 'NEG', 'NONE',
    'NOT SEEN', 'NOT DETECTED', '-', 'TRACE', '0',
}

# Fields that are qualitative (not float) in the main lab reports
SKIP_FIELDS = {
    'indirect_bilirubin',   # derived, redundant with total - direct
    'rbc_count',            # rarely available, adds noise
    'monocytes_pct',        # low predictive value for 8 classes
    'eosinophils_pct',
    'basophils_pct',
    'vldl',                 # derived from triglycerides
}


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def normalise_name(name):
    """Strip method suffixes and normalise spacing."""
    n = re.sub(r'\s*\(Method[^)]+\)', '', name, flags=re.IGNORECASE)
    n = re.sub(r'\s*\(Method-[^)]+\)', '', n, flags=re.IGNORECASE)
    n = re.sub(r'\s+', ' ', n).strip().upper()
    return n


def parse_numeric_value(val_str, field=None):
    """
    Extract the first numeric value from a string.
    Returns float or None. Handles ratio strings like '7.3:1' → 7.3.
    """
    if not val_str:
        return None
    # Handle ratio like "7.3:1" → take first number
    val_str = str(val_str).split(':')[0].strip()
    m = re.search(r'[\d,]+\.?\d*', val_str.replace(',', ''))
    if not m:
        return None
    try:
        v = float(m.group().replace(',', ''))
    except ValueError:
        return None
    # Platelet lakhs → absolute
    if field == 'platelets' and v < 100:
        v = round(v * 100000)
    return v


def parse_urine_value(val_str, field):
    """
    Parse a urine analysis value.
    Returns: 0 (negative/normal), 1 (abnormal/present), or float for numeric fields.
    """
    if not val_str:
        return None
    val = val_str.strip().upper()

    # Numeric fields (specific gravity, pH, pus cells etc.)
    if field in ('urine_specific_gravity', 'urine_ph',
                 'urine_pus_cells', 'urine_epithelial_cells', 'urine_rbc'):
        num = parse_numeric_value(val_str)
        return num

    # Qualitative fields
    if any(neg in val for neg in NEGATIVE_QUALITATIVE):
        return 0
    # 1+ / 2+ / 3+ / 4+ style
    m = re.search(r'(\d)\+', val)
    if m:
        return int(m.group(1))
    # PRESENT, POSITIVE, DETECTED etc.
    if any(pos in val for pos in ('PRESENT', 'POSITIVE', 'DETECTED')):
        return 1
    # Numeric fall-through
    num = parse_numeric_value(val_str)
    if num is not None:
        return num
    return None


def parse_ref_range(range_str):
    """
    Parse a reference range string → (low, high) or None.
    Handles: '136.0-145.0', '< 35', '> 6.5', '4.8 – 5.6 - Normal ...'
    """
    if not range_str:
        return None
    # Strip interpretation text after the range (e.g. "4.8 – 5.6 - Normal")
    s = range_str.split('-Normal')[0].split('- Normal')[0]
    s = s.split('- Pre')[0].split('-Pre')[0]
    s = s.replace('–', '-').replace('—', '-').strip()
    # < X
    m = re.match(r'^<\s*([\d.]+)', s)
    if m:
        return (0.0, float(m.group(1)))
    # > X
    m = re.match(r'^>\s*([\d.]+)', s)
    if m:
        return (float(m.group(1)), float('inf'))
    # X - Y  (but not ratio like 7.3:1)
    m = re.match(r'^([\d.]+)\s*-\s*([\d.]+)$', s.strip())
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo < hi:   # sanity check
            return (lo, hi)
    return None


def parse_date(date_str):
    """Extract datetime from any string containing dd-mm-yyyy."""
    if not date_str:
        return None
    m = re.search(r'(\d{2}-\d{2}-\d{4})', date_str)
    if m:
        try:
            return datetime.strptime(m.group(1), '%d-%m-%Y')
        except ValueError:
            pass
    return None


def extract_demographics(lines):
    """Extract age, gender, patient_type, uhid from the report header."""
    age, gender, patient_type, uhid = None, None, None, None
    for l in lines:
        # Age/Gender line like ":65Y / Male"
        m = re.search(r':?\s*(\d{1,3})\s*[Yy].*?/\s*(Male|Female|M|F)', l, re.I)
        if m and age is None:
            age_val = int(m.group(1))
            if 0 < age_val < 120:
                age = age_val
                gender_str = m.group(2).upper()
                gender = 1 if gender_str.startswith('M') else 0
        # Patient type
        if 'IPD' in l:
            patient_type = 'IPD'
        elif 'OPD' in l:
            patient_type = 'OPD'
        # UHID
        m2 = re.search(r'KIMS/(\d+)', l)
        if m2 and uhid is None:
            uhid = m2.group(1)
    return age, gender, patient_type, uhid


# ═══════════════════════════════════════════════════════════════════
#  SINGLE HTML REPORT PARSER
# ═══════════════════════════════════════════════════════════════════

def parse_one_html(fpath):
    """
    Parse one Srishti HTML report.

    Returns dict with:
      department, sample_date, sample_type,
      lab_values  {field: float},
      lab_ranges  {field: (lo, hi)},
      urine_values {field: int/float},
      peripheral_smear_impression: str or None,
      demographics {age, gender, patient_type, uhid}
    """
    soup  = BeautifulSoup(
        Path(fpath).read_text(encoding='utf-8', errors='ignore'), 'html.parser')
    text  = soup.get_text(separator='\n', strip=True)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    result = {
        'department':     None,
        'sample_date':    None,
        'sample_type':    None,
        'lab_values':     {},
        'lab_ranges':     {},
        'urine_values':   {},
        'smear':          None,
        'demographics':   {},
    }

    # ── Header info ────────────────────────────────────────────────
    header_end = 0
    for i, l in enumerate(lines):
        if 'DEPARTMENT OF' in l.upper() and result['department'] is None:
            result['department'] = l.strip()
        if 'Sampled On' in l:
            cand = l.split(':', 1)[-1].strip()
            if not cand and i+1 < len(lines):
                cand = lines[i+1]
            result['sample_date'] = parse_date(cand)
        if 'Sample Type' in l and i+1 < len(lines):
            nxt = lines[i+1].strip(': ')
            if nxt and len(nxt) < 30:
                result['sample_type'] = nxt
        if 'Test Name' in l or 'Biological Reference Interval' in l:
            header_end = i + 1
            break

    # Demographics from first 25 lines
    age, gender, pt, uhid = extract_demographics(lines[:25])
    result['demographics'] = {'age': age, 'gender': gender,
                               'patient_type': pt, 'uhid': uhid}

    # ── Is this a urine report? ────────────────────────────────────
    is_urine = (result['sample_type'] and 'URINE' in result['sample_type'].upper()) \
               or any('URINE ANALYSIS' in l.upper() for l in lines[header_end:header_end+5])

    if is_urine:
        result['urine_values'] = _parse_urine_section(lines, header_end)
        return result

    # ── Standard numeric lab report ───────────────────────────────
    noise = {'END OF THE REPORT', 'Prepared By', 'Verified By', 'APPROVED BY',
             'Srishti', 'Lab Reports', 'New Search', 'Page ', '©', 'PERIPHERAL SMEAR'}
    num_pat = re.compile(r'^[\d,]+\.?\d*$')

    i = header_end
    smear_lines = []
    in_smear = False
    while i < len(lines):
        l = lines[i]

        # Stop at noise markers
        if any(n in l for n in noise):
            if 'PERIPHERAL SMEAR' in l:
                in_smear = True
            elif in_smear and l.startswith(('--', '©', 'Prepared')):
                in_smear = False
            i += 1
            continue

        # Collect peripheral smear text
        if in_smear:
            smear_lines.append(l)
            i += 1
            continue

        # Check for IMPRESSION line (can appear anywhere after smear)
        if l.upper() == 'IMPRESSION':
            if i+1 < len(lines):
                result['smear'] = lines[i+1]
            i += 2
            continue

        # Standard 4-line block: TestName / Value / Unit / Range
        if i+1 < len(lines) and num_pat.match(lines[i+1].replace(',', '')):
            raw_name  = l
            raw_val   = lines[i+1]
            raw_unit  = lines[i+2] if i+2 < len(lines) else ''
            raw_range = lines[i+3] if i+3 < len(lines) else ''

            # Sometimes range is at i+2 (no unit line)
            if not parse_ref_range(raw_range) and parse_ref_range(raw_unit):
                raw_range = raw_unit

            norm  = normalise_name(raw_name)
            field = LAB_TEST_MAP.get(norm)

            if field and field not in SKIP_FIELDS:
                val = parse_numeric_value(raw_val, field)
                if val is not None:
                    rng = parse_ref_range(raw_range)
                    # Platelet range lakhs conversion
                    if field == 'platelets' and rng and rng[1] < 100:
                        rng = (rng[0]*100000, rng[1]*100000)
                    # Only store first occurrence (per HTML file)
                    if field not in result['lab_values']:
                        result['lab_values'][field] = val
                        if rng:
                            result['lab_ranges'][field] = rng
            i += 2
        else:
            i += 1

    if smear_lines:
        result['smear'] = ' '.join(smear_lines)

    return result


def _parse_urine_section(lines, start):
    """
    Parse a urine analysis report section.
    Returns dict of urine_field → value.
    """
    urine = {}
    noise = {'END OF THE REPORT', 'Prepared By', 'Verified By',
             'APPROVED BY', 'Srishti', '©', 'MICROSCOPIC'}
    # Skip header noise and section headers
    section_headers = {
        'URINE ANALYSIS(BIOCHEMICAL AND MICROSCOPY)',
        'CHEMICAL EXAMINATION OF URINE(AUTOMATED ANALYSER)',
        'MICROSCOPIC EXAMINATION (AUTOMATED ANALYSER)',
        'PHYSICAL EXAMINATION',
        'VOLUME', 'COLOUR', 'APPEARANCE',   # skip physical exam lines
    }

    # Urine test-value pairs are directly: TestName / Value (no units/range)
    # Some have numeric values, most have qualitative strings
    i = start
    while i < len(lines):
        l = lines[i]
        if any(n in l for n in noise):
            break
        norm = normalise_name(l)
        field = URINE_TEST_MAP.get(norm)
        if field and i+1 < len(lines):
            val_str = lines[i+1].strip()
            # Skip section headers as values
            if val_str.upper() not in section_headers and len(val_str) < 30:
                val = parse_urine_value(val_str, field)
                if val is not None and field not in urine:
                    urine[field] = val
            i += 2
        else:
            i += 1
    return urine


# ═══════════════════════════════════════════════════════════════════
#  PATIENT FOLDER MERGER
# ═══════════════════════════════════════════════════════════════════

def parse_patient_folder(folder):
    """
    Parse ALL HTML reports in a patient folder and merge intelligently.

    MERGING STRATEGY:
    - Demographics: taken from first report that has them
    - Admission window: earliest sample date ± 1 day
    - Lab values: admission-window values preferred; within window,
      first occurrence wins (consistent with clinical practice of
      using admission labs)
    - Timeline: ALL dated values stored for every field (audit trail)
    - Urine: merged across all urine reports
    - Peripheral smear: first non-null smear impression kept

    Returns: (result_dict, error_string)
    """
    folder = Path(folder)
    htmls  = sorted(folder.glob('*.html'))
    if not htmls:
        return None, 'no HTML files'

    # Parse all reports
    reports = []
    for h in htmls:
        try:
            r = parse_one_html(h)
            r['_source_file'] = h.name
            reports.append(r)
        except Exception as e:
            pass  # skip unreadable files silently

    # Filter to reports with actual values
    reports = [r for r in reports if r['lab_values'] or r['urine_values']]
    if not reports:
        return None, 'no parseable values in any report'

    # ── Demographics (from first report that has age) ──────────────
    demographics = {}
    for r in reports:
        d = r.get('demographics', {})
        if d.get('age') and not demographics.get('age'):
            demographics = d
    if not demographics:
        demographics = {'age': None, 'gender': None, 'patient_type': None, 'uhid': None}

    # ── Determine admission window ─────────────────────────────────
    dated_lab_reports = [r for r in reports if r['sample_date'] and r['lab_values']]
    if dated_lab_reports:
        earliest = min(r['sample_date'] for r in dated_lab_reports)
        admission_window = [
            r for r in dated_lab_reports
            if abs((r['sample_date'] - earliest).days) <= 1
        ]
    else:
        admission_window = [r for r in reports if r['lab_values']]
        earliest = None

    # ── Build timeline: ALL values across all dates ────────────────
    # timeline[field] = [(date, value, source_file), ...]
    timeline = defaultdict(list)
    for r in dated_lab_reports:
        dt = r['sample_date']
        for f, v in r['lab_values'].items():
            timeline[f].append({
                'date':  dt.strftime('%Y-%m-%d') if dt else None,
                'value': v,
                'file':  r['_source_file'],
            })

    # ── Merge labs: admission window first, then fill from later ──
    merged_labs   = {}
    merged_ranges = {}
    date_used     = {}   # which date each field came from

    for r in admission_window:
        for f, v in r['lab_values'].items():
            if f not in merged_labs:
                merged_labs[f]   = v
                date_used[f]     = r['sample_date'].strftime('%Y-%m-%d') \
                                   if r['sample_date'] else 'unknown'
                if f in r['lab_ranges']:
                    merged_ranges[f] = r['lab_ranges'][f]

    # Fill remaining fields from non-admission reports (better than missing)
    for r in reports:
        if r in admission_window:
            continue
        for f, v in r['lab_values'].items():
            if f not in merged_labs:
                merged_labs[f]   = v
                date_used[f]     = r['sample_date'].strftime('%Y-%m-%d') \
                                   if r.get('sample_date') else 'other_date'
                if f in r.get('lab_ranges', {}):
                    merged_ranges[f] = r['lab_ranges'][f]

    # ── Merge urine reports ────────────────────────────────────────
    merged_urine = {}
    for r in reports:
        for f, v in r.get('urine_values', {}).items():
            if f not in merged_urine:
                merged_urine[f] = v

    # ── Peripheral smear ───────────────────────────────────────────
    smear = next((r['smear'] for r in reports if r.get('smear')), None)

    # ── Audit: which fields came from which dates ──────────────────
    # Sort timeline by date for each field
    for f in timeline:
        timeline[f].sort(key=lambda x: x['date'] or '')

    return {
        'patient_id':         folder.name,
        'demographics':       demographics,
        'lab_features':       merged_labs,
        'lab_printed_ranges': {k: list(v) for k, v in merged_ranges.items()},
        'urine_features':     merged_urine,
        'lab_timeline':       {k: v for k, v in timeline.items()},
        'lab_date_used':      date_used,
        'peripheral_smear':   smear,
        'admission_date':     earliest.strftime('%Y-%m-%d') if earliest else None,
        'n_reports':          len([r for r in reports if r['lab_values']]),
        'n_labs':             len(merged_labs),
        'n_urine_fields':     len(merged_urine),
    }, None


# ═══════════════════════════════════════════════════════════════════
#  BATCH PARSER
# ═══════════════════════════════════════════════════════════════════

def parse_all(root_dir, out_dir):
    """Parse every patient folder under root_dir, write JSONs to out_dir."""
    from pathlib import Path
    import json
 
    try:
        from tqdm import tqdm
        _tqdm = True
    except ImportError:
        print("tqdm not found — pip install tqdm. Falling back to plain output.")
        _tqdm = False
 
    root = Path(root_dir)
    out  = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    folders = sorted([d for d in root.iterdir() if d.is_dir()])
    stats   = {
        'total':        len(folders),
        'success':      0,
        'errors':       0,
        'labs_total':   0,
        'urine_found':  0,
        'error_detail': [],
    }
 
    iterator = tqdm(
        folders,
        desc    = "Parsing patients",
        unit    = "pt",
        colour  = "cyan",
        dynamic_ncols = True,
        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "✓{postfix}"
        ),
    ) if _tqdm else folders
 
    for folder in iterator:
        result, err = parse_patient_folder(folder)
 
        if err:
            stats['errors'] += 1
            stats['error_detail'].append({'patient': folder.name, 'error': err})
        else:
            stats['success']    += 1
            stats['labs_total'] += result['n_labs']
            if result['n_urine_fields'] > 0:
                stats['urine_found'] += 1
 
            out_path = out / f"{folder.name}.json"
            out_path.write_text(
                json.dumps(result, indent=2, default=str), encoding='utf-8'
            )
 
        # Update tqdm postfix with live stats
        if _tqdm:
            iterator.set_postfix(
                ok    = stats['success'],
                err   = stats['errors'],
                labs  = f"{stats['labs_total'] // max(stats['success'], 1)} avg",
                urine = stats['urine_found'],
            )
 
    stats['avg_labs']    = round(stats['labs_total'] / max(stats['success'], 1), 1)
    stats['error_detail'] = stats['error_detail'][:20]
 
    (out / '_parse_summary.json').write_text(
        json.dumps(stats, indent=2), encoding='utf-8'
    )
 
    # Final summary line
    print(f"\n{'='*55}")
    print(f"  PARSE COMPLETE")
    print(f"{'='*55}")
    print(f"  Parsed:            {stats['success']} / {stats['total']}")
    print(f"  Errors:            {stats['errors']}")
    print(f"  Avg labs/patient:  {stats['avg_labs']}")
    print(f"  With urine data:   {stats['urine_found']}")
    print(f"  Output:            {out}")
    return stats



# ═══════════════════════════════════════════════════════════════════
#  CLI / TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 3:
        parse_all(sys.argv[1], sys.argv[2])
    else:
        # Test on the sample patient
        sample = Path('/home/claude/sample_reports/KIMS_202301127345')
        if not sample.exists():
            print("Usage: python parse_unlabelled_v2.py <root_dir> <out_dir>")
            sys.exit(0)

        result, err = parse_patient_folder(sample)
        if err:
            print(f"Error: {err}"); sys.exit(1)

        print(f"Patient:      {result['patient_id']}")
        d = result['demographics']
        print(f"Demographics: age={d['age']}  gender={'M' if d['gender']==1 else 'F'}  "
              f"type={d['patient_type']}  uhid={d['uhid']}")
        print(f"Admission:    {result['admission_date']}")
        print(f"Reports merged: {result['n_reports']}")
        print(f"Labs:  {result['n_labs']}    Urine fields: {result['n_urine_fields']}")
        print()

        print(f"{'FIELD':<22} {'VALUE':>10}   {'DATE USED':<12}  PRINTED RANGE")
        print('-' * 65)
        for f, v in sorted(result['lab_features'].items()):
            date = result['lab_date_used'].get(f, '?')
            rng  = result['lab_printed_ranges'].get(f, '—')
            tl   = result['lab_timeline'].get(f, [])
            multi = f"  [{len(tl)} dates]" if len(tl) > 1 else ''
            print(f"  {f:<20} {v:>10}   {date:<12}  {rng}{multi}")

        if result['urine_features']:
            print(f"\nURINE ANALYSIS ({result['n_urine_fields']} fields):")
            print('-' * 40)
            for f, v in result['urine_features'].items():
                print(f"  {f:<28} {v}")

        if result['peripheral_smear']:
            print(f"\nPERIPHERAL SMEAR: {result['peripheral_smear']}")

        # Show timeline conflicts
        print("\nFIELDS WITH MULTIPLE DATES (temporal changes):")
        for f, entries in result['lab_timeline'].items():
            if len(entries) > 1:
                vals = [(e['date'], e['value']) for e in entries]
                print(f"  {f:<20} {vals}")