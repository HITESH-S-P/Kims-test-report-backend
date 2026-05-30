r"""
train_component1.py  — v2
=========================
Changes from v1:
  - OneCycleLR scheduler with 10% warmup
  - LR raised to 1e-3, weight decay to 5e-4
  - Dropout raised to 0.4
  - Inverse-sqrt class weights (less aggressive)
  - Label smoothing 0.1 on condition loss
  - All 5 fold models saved + ensembled at inference
  - Rich markup fix ([bold green] instead of dynamic empty style)
"""

import json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from features import (extract_features, extract_labels, get_feature_dim,
                      CONDITION_TO_IDX, IDX_TO_CONDITION,
                      IDX_TO_SPECIALIST, ORGAN_FIELDS)
from model import Component1, Component1Loss

# ── Config ────────────────────────────────────────────────────────
COMBINED_DIR = Path(r'D:\Major_Project\project\kims_v3\data\combined')
MODEL_DIR    = Path(r'D:\Major_Project\project\kims_v3\models\component1')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS       = 200
BATCH_SIZE   = 32
LR           = 1e-3        # raised from 3e-4
WEIGHT_DECAY = 5e-4        # raised from 1e-4
N_FOLDS      = 5
DROPOUT      = 0.4         # raised from 0.3
PATIENCE     = 15


# ── Dataset ───────────────────────────────────────────────────────
class ClinicalDataset(Dataset):
    def __init__(self, patients):
        self.X      = []
        self.labels = []
        for p in patients:
            try:
                self.X.append(extract_features(p))
                self.labels.append(extract_labels(p))
            except:
                continue
        self.X = np.array(self.X, dtype=np.float32)

    def __len__(self):  return len(self.X)

    def __getitem__(self, idx):
        lb = self.labels[idx]
        return {
            'features':       torch.tensor(self.X[idx],          dtype=torch.float32),
            'condition_idx':  torch.tensor(lb['condition_idx'],   dtype=torch.long),
            'risk_idx':       torch.tensor(lb['risk_idx'],        dtype=torch.long),
            'specialist_idx': torch.tensor(lb['specialist_idx'],  dtype=torch.long),
            'organ_vec':      torch.tensor(lb['organ_vec'],       dtype=torch.float32),
            'weight':         torch.tensor(lb['confidence'],      dtype=torch.float32),
        }


# ── Load patients ─────────────────────────────────────────────────
def load_patients(data_dir):
    patients, errors = [], 0
    for fpath in sorted(data_dir.glob('*.json')):
        if fpath.name.startswith('_'):
            continue
        try:
            p = json.loads(fpath.read_text(encoding='utf-8'))
            if p.get('primary_condition') in CONDITION_TO_IDX:
                patients.append(p)
        except:
            errors += 1
    return patients, errors


# ── Inverse-sqrt class weights ────────────────────────────────────
def get_class_weights(patients):
    counts  = Counter(p['primary_condition'] for p in patients)
    total   = sum(counts.values())
    n_cls   = len(CONDITION_TO_IDX)
    weights = torch.ones(n_cls)
    for cond, idx in CONDITION_TO_IDX.items():
        c = counts.get(cond, 1)
        weights[idx] = (total / (n_cls * c)) ** 0.5   # sqrt — less aggressive
    return weights.to(DEVICE)


# ── Train one epoch ───────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, n_batches = 0, 0
    for batch in loader:
        features = batch['features'].to(DEVICE)
        labels   = {k: batch[k].to(DEVICE) for k in
                    ['condition_idx','risk_idx','specialist_idx','organ_vec']}
        weights  = batch['weight'].to(DEVICE)

        optimizer.zero_grad()
        loss_dict = criterion(model(features), labels, weights)
        loss_dict['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()          # OneCycleLR steps per batch

        total_loss += loss_dict['total'].item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


# ── Evaluate ──────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    cond_true, cond_pred = [], []
    risk_true, risk_pred = [], []
    org_true,  org_pred  = [], []

    with torch.no_grad():
        for batch in loader:
            out = model.predict(batch['features'].to(DEVICE))
            cond_true.extend(batch['condition_idx'].numpy())
            cond_pred.extend(out['condition_idx'].cpu().numpy())
            risk_true.extend(batch['risk_idx'].numpy())
            risk_pred.extend(out['risk_idx'].cpu().numpy())
            org_true.append(batch['organ_vec'].numpy())
            org_pred.append(out['organ_binary'].cpu().numpy())

    cf1  = f1_score(cond_true, cond_pred, average='macro', zero_division=0)
    racc = np.mean(np.array(risk_true) == np.array(risk_pred))
    of1  = f1_score(np.vstack(org_true), np.vstack(org_pred),
                    average='macro', zero_division=0)
    return {
        'condition_f1': round(cf1,  4),
        'risk_acc':     round(racc, 4),
        'organ_f1':     round(of1,  4),
        'combined':     round((cf1 + racc + of1) / 3, 4),
    }


# ── Ensemble evaluate (all saved fold models) ─────────────────────
def evaluate_ensemble(fold_states, input_dim, loader):
    """Average softmax/sigmoid outputs across all fold models."""
    models = []
    for state in fold_states:
        m = Component1(input_dim=input_dim, dropout=DROPOUT).to(DEVICE)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    cond_true, risk_true, org_true = [], [], []
    cond_probs_all, risk_probs_all, org_probs_all = [], [], []

    with torch.no_grad():
        for batch in loader:
            feats = batch['features'].to(DEVICE)
            cond_true.extend(batch['condition_idx'].numpy())
            risk_true.extend(batch['risk_idx'].numpy())
            org_true.append(batch['organ_vec'].numpy())

            cp_list, rp_list, op_list = [], [], []
            for m in models:
                out = m.predict(feats)
                cp_list.append(out['condition_probs'].cpu().numpy())
                rp_list.append(out['risk_probs'].cpu().numpy())
                op_list.append(out['organ_probs'].cpu().numpy())

            cond_probs_all.append(np.mean(cp_list, axis=0))
            risk_probs_all.append(np.mean(rp_list, axis=0))
            org_probs_all.append(np.mean(op_list,  axis=0))

    cond_pred = np.argmax(np.vstack(cond_probs_all), axis=1)
    risk_pred = np.argmax(np.vstack(risk_probs_all), axis=1)
    org_pred  = (np.vstack(org_probs_all) > 0.5).astype(int)

    cf1  = f1_score(cond_true, cond_pred, average='macro', zero_division=0)
    racc = np.mean(np.array(risk_true) == risk_pred)
    of1  = f1_score(np.vstack(org_true), org_pred, average='macro', zero_division=0)

    return {
        'condition_f1': round(cf1,  4),
        'risk_acc':     round(racc, 4),
        'organ_f1':     round(of1,  4),
        'combined':     round((cf1 + racc + of1) / 3, 4),
    }


# ══════════════════════════════════════════════════════════════════
#  MAIN
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

    state = {
        'status': 'Initializing…', 'fold': 0, 'epoch': 0,
        'metrics': {}, 'fold_results': [], 'best_combined': 0.0,
        'log': [], 'epoch_history': [], 'start_time': time.time(),
    }

    def add_log(msg):
        ts = time.strftime('%H:%M:%S')
        state['log'].append(f"[dim]{ts}[/dim]  {msg}")

    # ── Panel builders ─────────────────────────────────────────────
    def make_header():
        elapsed = int(time.time() - state['start_time'])
        m, s = divmod(elapsed, 60)
        fp = int(max(0, state['fold']-1) / N_FOLDS * 40)
        ep = int(state['epoch'] / EPOCHS * 40)
        t = Text()
        t.append("  Component 1 Trainer  ", style="bold white on dark_green")
        t.append(f"  device: {DEVICE}  elapsed: {m:02d}:{s:02d}\n", style="dim")
        t.append(f"\n  Folds   [cyan]{'█'*fp}[/cyan][dim]{'░'*(40-fp)}[/dim]"
                 f"  fold {state['fold']}/{N_FOLDS}\n")
        t.append(f"  Epochs  [green]{'█'*ep}[/green][dim]{'░'*(40-ep)}[/dim]"
                 f"  epoch {state['epoch']}/{EPOCHS}")
        return Panel(t, box=box.ROUNDED, border_style="green")

    def make_metrics():
        m = state['metrics']
        if not m:
            return Panel(Text("Waiting for first eval…", style="dim"),
                         title="[dim]latest metrics[/dim]", box=box.ROUNDED, border_style="dim")
        def bar(v, c, w=20):
            f = int(v*w)
            return f"[{c}]{'█'*f}[/{c}][dim]{'░'*(w-f)}[/dim]"
        def col(v): return 'green' if v>=0.7 else ('yellow' if v>=0.5 else 'red')
        cf1,ra,of1,cb = m['condition_f1'],m['risk_acc'],m['organ_f1'],m['combined']
        t = Text()
        t.append(f"  Condition F1  {bar(cf1,col(cf1))}  [{col(cf1)}]{cf1:.4f}[/{col(cf1)}]\n")
        t.append(f"  Risk Acc      {bar(ra, col(ra ))}  [{col(ra )}]{ra:.4f}[/{col(ra )}]\n")
        t.append(f"  Organ F1      {bar(of1,col(of1))}  [{col(of1)}]{of1:.4f}[/{col(of1)}]\n")
        t.append(f"  ─────────────────────────────────────────\n")
        t.append(f"  Combined      {bar(cb, col(cb ))}  [{col(cb )}]{cb:.4f}[/{col(cb )}]  ")
        t.append(f"best={state['best_combined']:.4f}", style="dim")
        return Panel(t, title="[dim]latest metrics[/dim]", box=box.ROUNDED, border_style="dim")

    def make_fold_table():
        if not state['fold_results']:
            return Panel(Text("No folds complete yet.", style="dim"),
                         title="[dim]fold summary[/dim]", box=box.ROUNDED, border_style="dim")
        tbl = Table(box=None, show_header=True, padding=(0,2), header_style="dim")
        for col_name, w in [("Fold",5),("Cond F1",9),("Risk Acc",9),("Organ F1",9),("Combined",9)]:
            tbl.add_column(col_name, width=w, justify="right")
        best_cb = max(f['combined'] for f in state['fold_results'])
        for i, fr in enumerate(state['fold_results']):
            is_best = fr['combined'] == best_cb
            # ── Rich fix: never use dynamic empty style tag ──────────
            cb_cell = (f"[bold green]{fr['combined']:.4f}[/bold green]"
                       if is_best else f"{fr['combined']:.4f}")
            tbl.add_row(
                f"{'★' if is_best else ' '}{i+1}",
                f"{fr['condition_f1']:.4f}",
                f"{fr['risk_acc']:.4f}",
                f"{fr['organ_f1']:.4f}",
                cb_cell,
            )
        if len(state['fold_results']) > 1:
            keys = ['condition_f1','risk_acc','organ_f1','combined']
            avgs = {k: np.mean([f[k] for f in state['fold_results']]) for k in keys}
            tbl.add_row("[dim]avg[/dim]",
                        *[f"[dim]{avgs[k]:.4f}[/dim]" for k in keys])
        return Panel(tbl, title="[dim]fold summary[/dim]", box=box.ROUNDED, border_style="dim")

    def make_log_panel():
        t = Text()
        for msg in state['log'][-14:]:
            t.append_text(Text.from_markup(f"{msg}\n"))
        return Panel(t, title=f"[dim]live log ({len(state['log'])} lines)[/dim]",
                     box=box.ROUNDED, border_style="dim", height=18)

    def make_sparkline():
        history = state['epoch_history']
        if not history:
            return Panel(Text("No loss data yet.", style="dim"),
                         title="[dim]loss curve[/dim]", box=box.ROUNDED, border_style="dim")
        losses = [h[1] for h in history]
        lo, hi = min(losses), max(losses)
        rng = hi - lo if hi != lo else 1
        bars  = '▁▂▃▄▅▆▇█'
        spark = ''.join(bars[int((v-lo)/rng*7)] for v in losses[-50:])
        t = Text()
        t.append(f"  [dim]{spark}[/dim]\n")
        t.append(f"  latest: [bold]{losses[-1]:.4f}[/bold]  "
                 f"min: [green]{lo:.4f}[/green]  max: [red]{hi:.4f}[/red]")
        return Panel(t, title="[dim]loss curve (last 50 evals)[/dim]",
                     box=box.ROUNDED, border_style="dim")

    def make_layout():
        layout = Layout()
        layout.split_column(
            Layout(make_header(),    size=7),
            Layout(Text(state['status'], style="bold cyan"), size=3),
            Layout(name="mid",       size=10),
            Layout(make_sparkline(), size=5),
            Layout(make_log_panel(), size=18),
        )
        layout['mid'].split_row(Layout(make_metrics()), Layout(make_fold_table()))
        return layout

    # ── Training ───────────────────────────────────────────────────
    if _rich:
        console  = Console()
        live_ctx = Live(make_layout(), refresh_per_second=8,
                        console=console, screen=False)
    else:
        import contextlib
        live_ctx = contextlib.nullcontext()

    def refresh():
        if _rich: live_ctx.update(make_layout())

    all_fold_states = []   # save ALL fold models for ensemble

    with live_ctx:
        state['status'] = 'Loading patients…'
        refresh()

        patients, load_errs = load_patients(COMBINED_DIR)
        input_dim = get_feature_dim()
        class_w   = get_class_weights(patients)

        add_log(f"[green]✓[/green] Loaded [bold]{len(patients)}[/bold] patients  "
                f"([dim]{load_errs} errors[/dim])")
        add_log(f"[cyan]→[/cyan] Input dim: [bold]{input_dim}[/bold]  device: [bold]{DEVICE}[/bold]")
        add_log(f"[cyan]→[/cyan] LR={LR}  dropout={DROPOUT}  weight_decay={WEIGHT_DECAY}  "
                f"label_smoothing=0.1  scheduler=OneCycleLR")
        refresh()

        labels_for_split = [CONDITION_TO_IDX[p['primary_condition']] for p in patients]
        skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        splits = list(skf.split(patients, labels_for_split))

        fold_results     = []
        best_overall     = 0
        best_model_state = None

        for fold, (train_idx, val_idx) in enumerate(splits):
            state['fold']    = fold + 1
            state['epoch']   = 0
            state['metrics'] = {}
            state['status']  = f"Fold {fold+1}/{N_FOLDS} — building datasets…"
            refresh()

            train_pts = [patients[i] for i in train_idx]
            val_pts   = [patients[i] for i in val_idx]
            train_ds  = ClinicalDataset(train_pts)
            val_ds    = ClinicalDataset(val_pts)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

            model     = Component1(input_dim=input_dim, dropout=DROPOUT).to(DEVICE)

            # ── Label smoothing added to condition loss ─────────────
            criterion = Component1Loss(condition_weights=class_w,
                                       label_smoothing=0.1)

            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=LR, weight_decay=WEIGHT_DECAY)

            # ── OneCycleLR: warmup 10% then cosine decay ────────────
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=LR,
                steps_per_epoch=len(train_loader),
                epochs=EPOCHS,
                pct_start=0.1,
                anneal_strategy='cos',
            )

            best_fold_score = 0
            best_fold_state = None
            patience_count  = 0

            add_log(f"[cyan]▶[/cyan] Fold [bold]{fold+1}[/bold] — "
                    f"train={len(train_pts)}  val={len(val_pts)}")

            for epoch in range(EPOCHS):
                state['epoch']  = epoch + 1
                state['status'] = f"Fold {fold+1}/{N_FOLDS} · Epoch {epoch+1}/{EPOCHS} · training…"
                refresh()

                train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    state['status'] = f"Fold {fold+1}/{N_FOLDS} · Epoch {epoch+1}/{EPOCHS} · eval…"
                    refresh()
                    metrics = evaluate(model, val_loader)
                    score   = metrics['combined']
                    state['metrics'] = metrics
                    state['epoch_history'].append((epoch+1, train_loss, score))

                    sc_col = 'green' if score >= 0.7 else ('yellow' if score >= 0.5 else 'red')
                    add_log(f"  [dim]f{fold+1} e{epoch+1:>3}[/dim]  "
                            f"loss=[bold]{train_loss:.4f}[/bold]  "
                            f"cond={metrics['condition_f1']}  "
                            f"risk={metrics['risk_acc']}  "
                            f"organ={metrics['organ_f1']}  "
                            f"[{sc_col}]→{score:.4f}[/{sc_col}]")

                    if score > best_fold_score:
                        best_fold_score = score
                        best_fold_state = {k: v.cpu().clone()
                                           for k, v in model.state_dict().items()}
                        patience_count  = 0
                        if score > state['best_combined']:
                            state['best_combined'] = score
                        add_log(f"[green]★[/green] New best: [bold green]{score:.4f}[/bold green]")
                    else:
                        patience_count += 1
                        if patience_count >= PATIENCE // 10:
                            add_log(f"[yellow]⚡[/yellow] Early stop fold {fold+1} @ epoch {epoch+1}")
                            break
                    refresh()

            # ── End of fold ────────────────────────────────────────
            model.load_state_dict(best_fold_state)
            final_metrics = evaluate(model, val_loader)
            fold_results.append(final_metrics)
            all_fold_states.append(best_fold_state)   # save for ensemble
            state['fold_results'] = fold_results
            state['metrics']      = final_metrics

            cb_col = 'green' if final_metrics['combined'] >= 0.7 else 'yellow'
            add_log(f"[green]✓[/green] Fold [bold]{fold+1}[/bold] done — "
                    f"cond={final_metrics['condition_f1']}  "
                    f"risk={final_metrics['risk_acc']}  "
                    f"organ={final_metrics['organ_f1']}  "
                    f"[{cb_col}]combined={final_metrics['combined']}[/{cb_col}]")

            # Save individual fold model
            torch.save({'model_state_dict': best_fold_state,
                        'input_dim': input_dim, 'dropout': DROPOUT},
                       MODEL_DIR / f'fold_{fold+1}_model.pt')

            if final_metrics['combined'] > best_overall:
                best_overall     = final_metrics['combined']
                best_model_state = best_fold_state
            refresh()

        # ── Ensemble evaluation on full dataset ────────────────────
        state['status'] = 'Computing ensemble metrics…'
        refresh()
        all_ds     = ClinicalDataset(patients)
        all_loader = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=False)
        ensemble_metrics = evaluate_ensemble(all_fold_states, input_dim, all_loader)
        add_log(f"[bold cyan]🔀 Ensemble[/bold cyan] — "
                f"cond={ensemble_metrics['condition_f1']}  "
                f"risk={ensemble_metrics['risk_acc']}  "
                f"organ={ensemble_metrics['organ_f1']}  "
                f"combined={ensemble_metrics['combined']}")
        refresh()

        # ── Save best single model + ensemble config ───────────────
        torch.save({
            'model_state_dict': best_model_state,
            'input_dim':        input_dim,
            'dropout':          DROPOUT,
            'config': {'epochs': EPOCHS, 'lr': LR,
                       'batch_size': BATCH_SIZE, 'n_folds': N_FOLDS,
                       'label_smoothing': 0.1, 'scheduler': 'OneCycleLR'},
        }, MODEL_DIR / 'best_model.pt')

        # Ensemble config — inference.py will load all fold models from this
        ensemble_cfg = {
            'fold_model_paths': [str(MODEL_DIR / f'fold_{i+1}_model.pt')
                                 for i in range(N_FOLDS)],
            'input_dim':  input_dim,
            'dropout':    DROPOUT,
            'n_folds':    N_FOLDS,
        }
        (MODEL_DIR / 'ensemble_config.json').write_text(
            json.dumps(ensemble_cfg, indent=2), encoding='utf-8'
        )

        avg = {k: round(np.mean([f[k] for f in fold_results]), 4) for k in fold_results[0]}
        std = {k: round(np.std( [f[k] for f in fold_results]), 4) for k in fold_results[0]}

        report = {
            'fold_results':     fold_results,
            'average':          avg,
            'std':              std,
            'ensemble_metrics': ensemble_metrics,
            'n_patients':       len(patients),
            'input_dim':        input_dim,
            'config': {
                'lr': LR, 'dropout': DROPOUT, 'weight_decay': WEIGHT_DECAY,
                'epochs': EPOCHS, 'label_smoothing': 0.1,
                'scheduler': 'OneCycleLR', 'class_weights': 'inv_sqrt',
            }
        }
        (MODEL_DIR / 'training_report.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8'
        )

        state['status'] = '✅  Training complete'
        add_log(f"[green]✓[/green] best_model.pt saved")
        add_log(f"[green]✓[/green] fold_1..{N_FOLDS}_model.pt saved")
        add_log(f"[green]✓[/green] ensemble_config.json saved")
        add_log(f"[bold green]✅  All done.[/bold green]")
        refresh()

    # ── Final summary ──────────────────────────────────────────────
    if _rich:
        console.print()
        console.print(f"[bold green]{'═'*62}[/bold green]")
        console.print(f"  [bold]TRAINING COMPLETE[/bold]")
        console.print(f"[bold green]{'═'*62}[/bold green]")
        console.print(f"  [dim]── Single best fold ──────────────────────────────[/dim]")
        console.print(f"  Condition F1:  [green]{avg['condition_f1']}[/green] ± {std['condition_f1']}")
        console.print(f"  Risk Accuracy: [green]{avg['risk_acc']}[/green]     ± {std['risk_acc']}")
        console.print(f"  Organ F1:      [green]{avg['organ_f1']}[/green]     ± {std['organ_f1']}")
        console.print(f"  Combined:      [bold]{avg['combined']}[/bold]     ± {std['combined']}")
        console.print(f"  [dim]── Ensemble (all {N_FOLDS} folds) ──────────────────────────[/dim]")
        console.print(f"  Condition F1:  [bold green]{ensemble_metrics['condition_f1']}[/bold green]")
        console.print(f"  Risk Accuracy: [bold green]{ensemble_metrics['risk_acc']}[/bold green]")
        console.print(f"  Organ F1:      [bold green]{ensemble_metrics['organ_f1']}[/bold green]")
        console.print(f"  Combined:      [bold green]{ensemble_metrics['combined']}[/bold green]")
        console.print(f"\n  Model dir → {MODEL_DIR}")
    else:
        print(f"\n{'='*62}")
        print(f"  TRAINING COMPLETE")
        print(f"{'='*62}")
        print(f"  ── Single best fold ──")
        print(f"  Condition F1:  {avg['condition_f1']} ± {std['condition_f1']}")
        print(f"  Risk Accuracy: {avg['risk_acc']}     ± {std['risk_acc']}")
        print(f"  Organ F1:      {avg['organ_f1']}     ± {std['organ_f1']}")
        print(f"  Combined:      {avg['combined']}     ± {std['combined']}")
        print(f"  ── Ensemble (all {N_FOLDS} folds) ──")
        print(f"  Condition F1:  {ensemble_metrics['condition_f1']}")
        print(f"  Risk Accuracy: {ensemble_metrics['risk_acc']}")
        print(f"  Organ F1:      {ensemble_metrics['organ_f1']}")
        print(f"  Combined:      {ensemble_metrics['combined']}")
        print(f"\n  Model dir: {MODEL_DIR}")


if __name__ == '__main__':
    train()