# Component 1 v2 — Semi-Supervised Retraining Pipeline

Retrains the KIMS clinical model using your 260 matched pairs + 59 discharge
summaries + the 2000 previously-unused lab reports, and adds a normal/healthy
detection layer with tiered responses.

## The 6 files and what they do

| File | Role |
|---|---|
| `lab_ranges.py` | Hardcoded clinical reference ranges + critical thresholds. Single source of truth. |
| `parse_unlabelled.py` | Parses the 2000 HTML reports → labs + printed reference ranges. |
| `build_semisupervised_data.py` | Turns 2000 reports into training data: deterministic normal cases + >85% confidence pseudo-labels. |
| `model_v2.py` | Component1V2 — adds 5th binary health head (healthy vs sick). |
| `train_component1_v2.py` | 5-fold training with health head, sick-recall metric, Rich UI. |
| `clinical_gate.py` | Inference gate: rule layer + health head + tiered normal response. |

## Run order

### Step 1 — Parse the 2000 reports
Point this at the folder containing all 2000 patient subfolders.
```powershell
conda activate med
cd D:\Major_Project\project\kims_v3\src\pipeline_v2

python parse_unlabelled.py ^
  "D:\path\to\2000_reports_root" ^
  "D:\Major_Project\project\kims_v3\data\unlabelled_parsed"
```

### Step 2 — Build semi-supervised training data
Copy `features.py` and `model.py` (v1) into this folder first, since
pseudo-labelling uses your CURRENT trained ensemble.
```powershell
copy ..\component1\features.py .
copy ..\component1\model.py .

python build_semisupervised_data.py ^
  "D:\Major_Project\project\kims_v3\data\unlabelled_parsed" ^
  "D:\Major_Project\project\kims_v3\data\semisupervised" ^
  "D:\Major_Project\project\kims_v3\models\component1"
```
This prints how many normal cases and pseudo-labels were extracted.

### Step 3 — Combine all training data into combined_v2
```powershell
mkdir D:\Major_Project\project\kims_v3\data\combined_v2
copy D:\Major_Project\project\kims_v3\data\combined\*.json        D:\Major_Project\project\kims_v3\data\combined_v2\
copy D:\Major_Project\project\kims_v3\data\semisupervised\*.json  D:\Major_Project\project\kims_v3\data\combined_v2\
```
(Existing combined/ already has matched pairs + discharge + augmented.
 We're adding the normal + pseudo-labelled cases on top.)

### Step 4 — Retrain
Copy `features.py` into pipeline_v2 if not already there.
```powershell
python train_component1_v2.py
```
Watch the **Sick Recall** metric — that's the safety number. It must stay
high (target >0.95). It answers: "of all genuinely sick patients, how many
did the model correctly flag as needing attention?"

### Step 5 — Wire the gate into inference
In your `inference.py` / `app.py`, before showing condition results:
```python
from clinical_gate import run_gate

gate = run_gate(labs, mask, gender_str, health_head_output=c1_preds)
if not gate['run_full_pipeline']:
    # Show tiered normal response, skip Mistral
    show_tiered(gate['tiered_response'])
else:
    # Run your existing Mistral verification + narrative
    run_mistral_pipeline(...)
```

## Key safety design choices

1. **Health head is SEPARATE, not a 9th class.** The 8-condition head is
   never diluted by normal examples, so your 0.847 condition F1 is preserved.

2. **Condition losses are masked on healthy patients.** A normal person has
   no condition, so we don't train the condition head on them.

3. **Sick recall is weighted highest.** A false "healthy" on a sick patient
   is the worst error. Health-head loss is 2x, with 1.5x extra weight on the
   sick class.

4. **Gate is safety-biased.** Full analysis runs if EITHER the rule layer OR
   the health head thinks the patient is sick. We over-investigate rather
   than miss.

5. **Pseudo-labels get low weight (0.60)** vs gold matched pairs (0.85), so
   model guesses never override real ground truth.

## Label weights summary
| Source | Weight | Trust |
|---|---|---|
| Matched pairs (real discharge labels) | 0.85 | Gold |
| Discharge summaries | 0.80 | High |
| Normal cases (deterministic rule) | 0.85 | High (rule, not guess) |
| Pseudo-labelled (>85% model conf) | 0.60 | Lower (model guess) |

## Tuning notes
- `PSEUDO_CONF_THRESHOLD = 0.85` in build_semisupervised_data.py — raise to
  0.90 for fewer/cleaner pseudo-labels.
- `SIGNIFICANCE_THRESHOLD = 2.0` in clinical_gate.py — controls when a patient
  with minor deviations gets full analysis vs tiered "mild" response.
- `MIN_CORE_LABS_FOR_NORMAL = 8` — a report needs ≥8 core labs measured to be
  eligible for the normal class (prevents calling a 2-lab report "healthy").
