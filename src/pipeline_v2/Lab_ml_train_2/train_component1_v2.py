r"""
train_component1_v2.py
======================
Retrains Component 1 with:
  - 5th binary health head (healthy vs sick)
  - matched pairs (weight 0.85) + discharge summaries (0.80)
    + pseudo-labelled from 2000 (0.60) + normal cases (0.85)
  - per-sample weighting (label_confidence drives trust)
  - inverse-sqrt class weights, label smoothing 0.1
  - OneCycleLR, 5-fold CV, ensemble
  - Rich live monitoring (same style as v1)

MEDICAL SAFETY:
  - health head weighted 2x (a false 'healthy' is the worst error)
  - condition heads masked on normal examples (don't pollute them)
  - we report health-head RECALL on sick patients explicitly — this is
    the number that matters: of all genuinely sick patients, how many
    did we correctly flag as needing attention.

HOW TO RUN:
    python train_component1_v2.py

Place this in src/component1/ alongside features.py + model_v2.py.
"""

import json, time, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score

from features import (extract_features, get_feature_dim,
                      CONDITION_TO_IDX, IDX_TO_CONDITION,
                      IDX_TO_SPECIALIST, ORGAN_FIELDS)
from model_v2 import Component1V2, Component1V2Loss

# ── Config ─────────────────────────────────────────────────────────
# Combined dir should contain ALL training JSONs:
#   matched pairs + discharge summaries + semisupervised (normal+pseudo)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed

COMBINED_DIR = PROJECT_ROOT / "data" / "combined_v2"
MODEL_DIR    = PROJECT_ROOT / "models" / "component1_v2"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS       = 200
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 5e-4
N_FOLDS      = 5
DROPOUT      = 0.4
PATIENCE     = 15


# ── Dataset ────────────────────────────────────────────────────────
class ClinicalDatasetV2(Dataset):
    def __init__(self, patients):
        self.X, self.labels = [], []
        for p in patients:
            try:
                self.X.append(extract_features(p))
                self.labels.append(self._extract_labels(p))
            except Exception:
                continue
        self.X = np.array(self.X, dtype=np.float32)

    def _extract_labels(self, p):
        tgt = p.get('targets', {})
        pc  = p.get('primary_condition', 'infection')
        is_healthy = int(p.get('is_healthy', 0))
        # For normal patients, condition_idx is a placeholder (masked in loss)
        cond_idx = CONDITION_TO_IDX.get(pc, 0) if pc in CONDITION_TO_IDX else 0
        risk_idx = max(0, min(int(tgt.get('risk_level', 2)) - 1, 3))
        spec_raw = int(tgt.get('specialist', 10))
        spec_idx = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9}.get(spec_raw, 9)
        org = p.get('organ_involvement', {})
        organ_vec = np.array([float(org.get(f, 0) or 0) for f in ORGAN_FIELDS],
                             dtype=np.float32)
        return {
            'condition_idx':  cond_idx,
            'risk_idx':       risk_idx,
            'specialist_idx': spec_idx,
            'organ_vec':      organ_vec,
            'is_healthy':     is_healthy,
            'confidence':     float(p.get('label_confidence', 0.8)),
        }

    def __len__(self):  return len(self.X)

    def __getitem__(self, idx):
        lb = self.labels[idx]
        return {
            'features':       torch.tensor(self.X[idx],         dtype=torch.float32),
            'condition_idx':  torch.tensor(lb['condition_idx'],  dtype=torch.long),
            'risk_idx':       torch.tensor(lb['risk_idx'],       dtype=torch.long),
            'specialist_idx': torch.tensor(lb['specialist_idx'], dtype=torch.long),
            'organ_vec':      torch.tensor(lb['organ_vec'],      dtype=torch.float32),
            'is_healthy':     torch.tensor(lb['is_healthy'],     dtype=torch.long),
            'weight':         torch.tensor(lb['confidence'],     dtype=torch.float32),
        }


def load_patients(data_dir):
    patients, errors = [], 0
    for fp in sorted(data_dir.glob('*.json')):
        if fp.name.startswith('_'):
            continue
        try:
            p = json.loads(fp.read_text(encoding='utf-8'))
            # Accept if it's a known condition OR a normal case
            if p.get('is_healthy') == 1 or p.get('primary_condition') in CONDITION_TO_IDX:
                patients.append(p)
        except Exception:
            errors += 1
    return patients, errors


def get_class_weights(patients):
    counts = Counter(p['primary_condition'] for p in patients
                     if p.get('is_healthy') != 1)
    total  = sum(counts.values())
    n_cls  = len(CONDITION_TO_IDX)
    w = torch.ones(n_cls)
    for cond, idx in CONDITION_TO_IDX.items():
        c = counts.get(cond, 1)
        w[idx] = (total / (n_cls * c)) ** 0.5
    return w.to(DEVICE)


def get_health_weights(patients):
    """Weight the health head. Penalise misclassifying SICK patients more."""
    n_healthy = sum(1 for p in patients if p.get('is_healthy') == 1)
    n_sick    = len(patients) - n_healthy
    # class 0 = not healthy (sick), class 1 = healthy
    # higher weight on class 0 so we rarely miss a sick patient
    total = n_healthy + n_sick
    w_sick    = total / (2 * max(n_sick, 1))
    w_healthy = total / (2 * max(n_healthy, 1))
    # extra safety multiplier on sick
    w_sick *= 1.5
    return torch.tensor([w_sick, w_healthy], dtype=torch.float32).to(DEVICE)


# ── Train / eval ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total, n = 0, 0
    for b in loader:
        feats  = b['features'].to(DEVICE)
        labels = {
            'condition_idx':  b['condition_idx'].to(DEVICE),
            'risk_idx':       b['risk_idx'].to(DEVICE),
            'specialist_idx': b['specialist_idx'].to(DEVICE),
            'organ_vec':      b['organ_vec'].to(DEVICE),
            'is_healthy':     b['is_healthy'].to(DEVICE),
        }
        w = b['weight'].to(DEVICE)
        optimizer.zero_grad()
        ld = criterion(model(feats), labels, w)
        ld['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += float(ld['total']); n += 1
    return total / max(n, 1)


def evaluate(model, loader):
    model.eval()
    ct, cp, rt, rp = [], [], [], []
    ot, op = [], []
    ht, hp = [], []          # health true / pred
    sw_mask = []             # which are sick
    with torch.no_grad():
        for b in loader:
            out = model.predict(b['features'].to(DEVICE))
            sick = (b['is_healthy'].numpy() == 0)
            # condition metrics only on sick patients
            for i in range(len(sick)):
                if sick[i]:
                    ct.append(int(b['condition_idx'][i]))
                    cp.append(int(out['condition_idx'][i].cpu()))
                    rt.append(int(b['risk_idx'][i]))
                    rp.append(int(out['risk_idx'][i].cpu()))
            ot.append(b['organ_vec'].numpy())
            op.append(out['organ_binary'].cpu().numpy())
            ht.extend(b['is_healthy'].numpy())
            hp.extend(out['health_idx'].cpu().numpy())

    cond_f1  = f1_score(ct, cp, average='macro', zero_division=0) if ct else 0.0
    risk_acc = np.mean(np.array(rt) == np.array(rp)) if rt else 0.0
    organ_f1 = f1_score(np.vstack(ot), np.vstack(op), average='macro', zero_division=0)

    ht, hp = np.array(ht), np.array(hp)
    health_acc = np.mean(ht == hp)
    # SICK RECALL = of all sick patients (label 0), how many did we catch?
    # In sklearn recall for class 0:
    sick_recall = recall_score(ht, hp, pos_label=0, zero_division=0)

    return {
        'condition_f1': round(float(cond_f1), 4),
        'risk_acc':     round(float(risk_acc), 4),
        'organ_f1':     round(float(organ_f1), 4),
        'health_acc':   round(float(health_acc), 4),
        'sick_recall':  round(float(sick_recall), 4),
        'combined':     round(float((cond_f1 + risk_acc + organ_f1 + sick_recall) / 4), 4),
    }


def evaluate_ensemble(states, input_dim, loader):
    models = []
    for st in states:
        m = Component1V2(input_dim=input_dim, dropout=DROPOUT).to(DEVICE)
        m.load_state_dict(st); m.eval()
        models.append(m)

    ct, cp, rt, rp, ot, op, ht, hp = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            feats = b['features'].to(DEVICE)
            sick  = (b['is_healthy'].numpy() == 0)
            cprobs, hprobs, oprobs, rprobs = [], [], [], []
            for m in models:
                o = m.predict(feats)
                cprobs.append(o['condition_probs'].cpu().numpy())
                hprobs.append(o['health_probs'].cpu().numpy())
                oprobs.append(o['organ_probs'].cpu().numpy())
                rprobs.append(o['risk_probs'].cpu().numpy())
            cpred = np.argmax(np.mean(cprobs, axis=0), axis=1)
            hpred = np.argmax(np.mean(hprobs, axis=0), axis=1)
            rpred = np.argmax(np.mean(rprobs, axis=0), axis=1)
            opred = (np.mean(oprobs, axis=0) > 0.5).astype(int)
            for i in range(len(sick)):
                if sick[i]:
                    ct.append(int(b['condition_idx'][i])); cp.append(int(cpred[i]))
                    rt.append(int(b['risk_idx'][i]));      rp.append(int(rpred[i]))
            ot.append(b['organ_vec'].numpy()); op.append(opred)
            ht.extend(b['is_healthy'].numpy()); hp.extend(hpred)

    cond_f1  = f1_score(ct, cp, average='macro', zero_division=0) if ct else 0.0
    risk_acc = np.mean(np.array(rt) == np.array(rp)) if rt else 0.0
    organ_f1 = f1_score(np.vstack(ot), np.vstack(op), average='macro', zero_division=0)
    ht, hp = np.array(ht), np.array(hp)
    health_acc  = np.mean(ht == hp)
    sick_recall = recall_score(ht, hp, pos_label=0, zero_division=0)
    return {
        'condition_f1': round(float(cond_f1), 4),
        'risk_acc':     round(float(risk_acc), 4),
        'organ_f1':     round(float(organ_f1), 4),
        'health_acc':   round(float(health_acc), 4),
        'sick_recall':  round(float(sick_recall), 4),
        'combined':     round(float((cond_f1+risk_acc+organ_f1+sick_recall)/4), 4),
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN with Rich
# ══════════════════════════════════════════════════════════════════
def train():
    try:
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.console import Console
        from rich import box
        _rich = True
    except ImportError:
        _rich = False

    state = {'status':'Init…','fold':0,'epoch':0,'metrics':{},
             'fold_results':[],'best':0.0,'log':[],'hist':[],'t0':time.time()}

    def log(msg):
        state['log'].append(f"[dim]{time.strftime('%H:%M:%S')}[/dim]  {msg}")

    def header():
        el = int(time.time()-state['t0']); m,s = divmod(el,60)
        fp = int(max(0,state['fold']-1)/N_FOLDS*40); ep=int(state['epoch']/EPOCHS*40)
        t=Text()
        t.append("  Component 1 v2 Trainer  ", style="bold white on dark_green")
        t.append(f"  {DEVICE}  {m:02d}:{s:02d}\n", style="dim")
        t.append(f"\n  Folds   [cyan]{'█'*fp}[/cyan][dim]{'░'*(40-fp)}[/dim]  fold {state['fold']}/{N_FOLDS}\n")
        t.append(f"  Epochs  [green]{'█'*ep}[/green][dim]{'░'*(40-ep)}[/dim]  epoch {state['epoch']}/{EPOCHS}")
        return Panel(t, box=box.ROUNDED, border_style="green")

    def metrics_panel():
        m=state['metrics']
        if not m:
            return Panel(Text("waiting…",style="dim"),title="[dim]metrics[/dim]",box=box.ROUNDED,border_style="dim")
        def bar(v,c,w=18):
            f=int(v*w); return f"[{c}]{'█'*f}[/{c}][dim]{'░'*(w-f)}[/dim]"
        def col(v): return 'green' if v>=0.7 else ('yellow' if v>=0.5 else 'red')
        t=Text()
        t.append(f"  Condition F1  {bar(m['condition_f1'],col(m['condition_f1']))}  {m['condition_f1']}\n")
        t.append(f"  Risk Acc      {bar(m['risk_acc'],col(m['risk_acc']))}  {m['risk_acc']}\n")
        t.append(f"  Organ F1      {bar(m['organ_f1'],col(m['organ_f1']))}  {m['organ_f1']}\n")
        t.append(f"  Health Acc    {bar(m['health_acc'],col(m['health_acc']))}  {m['health_acc']}\n")
        sr_col = 'green' if m['sick_recall']>=0.95 else ('yellow' if m['sick_recall']>=0.85 else 'red')
        t.append(f"  [bold]Sick Recall[/bold]   {bar(m['sick_recall'],sr_col)}  [bold {sr_col}]{m['sick_recall']}[/bold {sr_col}]\n")
        t.append(f"  ─────────────────────────────────\n")
        t.append(f"  Combined      {bar(m['combined'],col(m['combined']))}  {m['combined']}")
        return Panel(t,title="[dim]metrics (sick-recall = safety)[/dim]",box=box.ROUNDED,border_style="dim")

    def fold_table():
        if not state['fold_results']:
            return Panel(Text("no folds yet",style="dim"),title="[dim]folds[/dim]",box=box.ROUNDED,border_style="dim")
        tb=Table(box=None,show_header=True,padding=(0,1),header_style="dim")
        for c,w in [("F",3),("Cond",7),("Risk",6),("Org",6),("Health",7),("SickRec",8),("Comb",7)]:
            tb.add_column(c,width=w,justify="right")
        best=max(f['combined'] for f in state['fold_results'])
        for i,fr in enumerate(state['fold_results']):
            isb=fr['combined']==best
            comb=f"[bold green]{fr['combined']:.3f}[/bold green]" if isb else f"{fr['combined']:.3f}"
            tb.add_row(f"{'★'if isb else ''}{i+1}",f"{fr['condition_f1']:.3f}",
                       f"{fr['risk_acc']:.3f}",f"{fr['organ_f1']:.3f}",
                       f"{fr['health_acc']:.3f}",f"{fr['sick_recall']:.3f}",comb)
        if len(state['fold_results'])>1:
            keys=['condition_f1','risk_acc','organ_f1','health_acc','sick_recall','combined']
            a={k:np.mean([f[k] for f in state['fold_results']]) for k in keys}
            tb.add_row("avg",*[f"[dim]{a[k]:.3f}[/dim]" for k in keys])
        return Panel(tb,title="[dim]fold summary[/dim]",box=box.ROUNDED,border_style="dim")

    def log_panel():
        t=Text()
        for m in state['log'][-13:]:
            t.append_text(Text.from_markup(f"{m}\n"))
        return Panel(t,title=f"[dim]log ({len(state['log'])})[/dim]",box=box.ROUNDED,border_style="dim",height=17)

    def layout():
        L=Layout()
        L.split_column(
            Layout(header(),size=7),
            Layout(Text(state['status'],style="bold cyan"),size=2),
            Layout(name="mid",size=11),
            Layout(log_panel(),size=17),
        )
        L['mid'].split_row(Layout(metrics_panel()),Layout(fold_table()))
        return L

    if _rich:
        console=Console()
        live=Live(layout(),refresh_per_second=8,console=console,screen=False)
    else:
        import contextlib; live=contextlib.nullcontext()
    def refresh():
        if _rich: live.update(layout())

    all_states=[]
    with live:
        state['status']='Loading patients…'; refresh()
        patients,errs=load_patients(COMBINED_DIR)
        input_dim=get_feature_dim()
        class_w =get_class_weights(patients)
        health_w=get_health_weights(patients)

        n_healthy=sum(1 for p in patients if p.get('is_healthy')==1)
        n_sick=len(patients)-n_healthy
        log(f"[green]✓[/green] Loaded [bold]{len(patients)}[/bold] patients ([dim]{errs} errors[/dim])")
        log(f"[cyan]→[/cyan] Sick: [bold]{n_sick}[/bold]  Healthy: [bold]{n_healthy}[/bold]")
        log(f"[cyan]→[/cyan] Health weights: sick={health_w[0]:.2f} healthy={health_w[1]:.2f}")
        log(f"[cyan]→[/cyan] dim={input_dim} LR={LR} dropout={DROPOUT} OneCycleLR label_smooth=0.1")
        refresh()

        # Stratify on a combined label: healthy flag dominates, else condition
        strat=[]
        for p in patients:
            if p.get('is_healthy')==1:
                strat.append(8)   # 'normal' bucket
            else:
                strat.append(CONDITION_TO_IDX[p['primary_condition']])
        skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=42)
        splits=list(skf.split(patients,strat))

        fold_results=[]; best_overall=0; best_state=None
        for fold,(tr,va) in enumerate(splits):
            state['fold']=fold+1; state['epoch']=0; state['metrics']={}
            state['status']=f"Fold {fold+1} — building…"; refresh()

            tr_pts=[patients[i] for i in tr]; va_pts=[patients[i] for i in va]
            tr_ds=ClinicalDatasetV2(tr_pts); va_ds=ClinicalDatasetV2(va_pts)
            tr_ld=DataLoader(tr_ds,batch_size=BATCH_SIZE,shuffle=True)
            va_ld=DataLoader(va_ds,batch_size=BATCH_SIZE,shuffle=False)

            model=Component1V2(input_dim=input_dim,dropout=DROPOUT).to(DEVICE)
            crit =Component1V2Loss(condition_weights=class_w,label_smoothing=0.1,
                                   health_weights=health_w)
            opt  =torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
            sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,
                    steps_per_epoch=len(tr_ld),epochs=EPOCHS,pct_start=0.1,
                    anneal_strategy='cos')

            best_fold=0; best_fs=None; patience=0
            log(f"[cyan]▶[/cyan] Fold [bold]{fold+1}[/bold] train={len(tr_pts)} val={len(va_pts)}")
            for ep in range(EPOCHS):
                state['epoch']=ep+1
                state['status']=f"Fold {fold+1} · Epoch {ep+1}/{EPOCHS} · training…"; refresh()
                loss=train_epoch(model,tr_ld,opt,sched,crit)
                if (ep+1)%10==0 or ep==0:
                    state['status']=f"Fold {fold+1} · Epoch {ep+1} · eval…"; refresh()
                    mt=evaluate(model,va_ld); sc=mt['combined']
                    state['metrics']=mt; state['hist'].append((ep+1,loss,sc))
                    scc='green' if sc>=0.7 else ('yellow' if sc>=0.5 else 'red')
                    log(f"  [dim]f{fold+1} e{ep+1:>3}[/dim] loss=[bold]{loss:.3f}[/bold] "
                        f"cond={mt['condition_f1']} health={mt['health_acc']} "
                        f"[bold]sickrec={mt['sick_recall']}[/bold] [{scc}]→{sc:.3f}[/{scc}]")
                    if sc>best_fold:
                        best_fold=sc
                        best_fs={k:v.cpu().clone() for k,v in model.state_dict().items()}
                        patience=0
                        if sc>state['best']: state['best']=sc
                        log(f"[green]★[/green] new best [bold green]{sc:.3f}[/bold green]")
                    else:
                        patience+=1
                        if patience>=PATIENCE//10:
                            log(f"[yellow]⚡[/yellow] early stop f{fold+1} @ e{ep+1}")
                            break
                    refresh()
            model.load_state_dict(best_fs)
            fm=evaluate(model,va_ld); fold_results.append(fm)
            all_states.append(best_fs); state['fold_results']=fold_results
            state['metrics']=fm
            log(f"[green]✓[/green] Fold [bold]{fold+1}[/bold] sickrec={fm['sick_recall']} comb={fm['combined']}")
            torch.save({'model_state_dict':best_fs,'input_dim':input_dim,'dropout':DROPOUT},
                       MODEL_DIR/f'fold_{fold+1}_model.pt')
            if fm['combined']>best_overall:
                best_overall=fm['combined']; best_state=best_fs
            refresh()

        # Ensemble
        state['status']='Ensemble eval…'; refresh()
        all_ds=ClinicalDatasetV2(patients)
        all_ld=DataLoader(all_ds,batch_size=BATCH_SIZE,shuffle=False)
        ens=evaluate_ensemble(all_states,input_dim,all_ld)
        log(f"[bold cyan]🔀 Ensemble[/bold cyan] cond={ens['condition_f1']} "
            f"health={ens['health_acc']} [bold]sickrec={ens['sick_recall']}[/bold] comb={ens['combined']}")
        refresh()

        torch.save({'model_state_dict':best_state,'input_dim':input_dim,
                    'dropout':DROPOUT,
                    'config':{'epochs':EPOCHS,'lr':LR,'label_smoothing':0.1,
                              'has_health_head':True}},
                   MODEL_DIR/'best_model.pt')
        (MODEL_DIR/'ensemble_config.json').write_text(json.dumps({
            'fold_model_paths':[str(MODEL_DIR/f'fold_{i+1}_model.pt') for i in range(N_FOLDS)],
            'input_dim':input_dim,'dropout':DROPOUT,'n_folds':N_FOLDS,
            'has_health_head':True,
        },indent=2),encoding='utf-8')

        avg={k:round(np.mean([f[k] for f in fold_results]),4) for k in fold_results[0]}
        std={k:round(np.std([f[k] for f in fold_results]),4) for k in fold_results[0]}
        (MODEL_DIR/'training_report.json').write_text(json.dumps({
            'fold_results':fold_results,'average':avg,'std':std,
            'ensemble_metrics':ens,'n_patients':len(patients),
            'n_sick':n_sick,'n_healthy':n_healthy,'input_dim':input_dim,
        },indent=2),encoding='utf-8')

        state['status']='✅ Complete'; log("[bold green]✅ done[/bold green]"); refresh()

    # Final summary
    if _rich:
        console.print()
        console.print(f"[bold green]{'═'*62}[/bold green]")
        console.print(f"  [bold]TRAINING COMPLETE — Component 1 v2[/bold]")
        console.print(f"[bold green]{'═'*62}[/bold green]")
        console.print(f"  [dim]── Single fold avg ──[/dim]")
        console.print(f"  Condition F1:  {avg['condition_f1']} ± {std['condition_f1']}")
        console.print(f"  Risk Acc:      {avg['risk_acc']} ± {std['risk_acc']}")
        console.print(f"  Organ F1:      {avg['organ_f1']} ± {std['organ_f1']}")
        console.print(f"  Health Acc:    {avg['health_acc']} ± {std['health_acc']}")
        console.print(f"  [bold]Sick Recall:   {avg['sick_recall']} ± {std['sick_recall']}[/bold]")
        console.print(f"  [dim]── Ensemble (all {N_FOLDS} folds) ──[/dim]")
        console.print(f"  Condition F1:  [bold green]{ens['condition_f1']}[/bold green]")
        console.print(f"  Health Acc:    [bold green]{ens['health_acc']}[/bold green]")
        console.print(f"  [bold]Sick Recall:   [bold green]{ens['sick_recall']}[/bold green][/bold]  ← most important")
        console.print(f"  Combined:      [bold green]{ens['combined']}[/bold green]")
        console.print(f"\n  Model dir → {MODEL_DIR}")
    else:
        print("TRAINING COMPLETE")
        print("avg:", avg)
        print("ensemble:", ens)


if __name__ == '__main__':
    train()
