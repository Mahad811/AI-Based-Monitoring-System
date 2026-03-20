"""Quick threshold sweep analysis on saved probability data."""
import json
import numpy as np

with open('seizure_detection/report_eval/test_raw_probs.json') as f:
    data = json.load(f)

seizure = [v for v in data if v['label'] == 1]
normal = [v for v in data if v['label'] == 0]
print(f"Total: {len(data)} videos ({len(seizure)} seizure, {len(normal)} normal)")

no_windows = [v for v in data if v['n_windows'] == 0]
print(f"Videos with 0 windows (person not detected): {len(no_windows)}")
for v in no_windows:
    print(f"  {v['name']} (label={v['label']})")

def agg(probs, method):
    if not probs:
        return 0.0
    if method == 'max':
        return max(probs)
    if method == 'mean':
        return float(np.mean(probs))
    if method == 'p90':
        return float(np.percentile(probs, 90))
    if method == 'top3':
        return float(np.mean(sorted(probs, reverse=True)[:3]))
    if method == 'top5':
        return float(np.mean(sorted(probs, reverse=True)[:5]))
    return max(probs)

def metrics(scores, labels, t):
    TP = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
    FP = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
    FN = sum(1 for s, l in zip(scores, labels) if s < t and l == 1)
    TN = sum(1 for s, l in zip(scores, labels) if s < t and l == 0)
    p = TP / (TP + FP) if TP + FP > 0 else 0
    r = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    return TP, FP, FN, TN, p, r, f1, acc

labels = [v['label'] for v in data]

print()
print('=' * 80)
print('THRESHOLD SWEEP RESULTS (all aggregation methods)')
print('=' * 80)

best_overall_f1 = 0
best_overall_config = None
best_medical_config = None
best_medical_precision = 0

for method in ['max', 'mean', 'p90', 'top3', 'top5']:
    scores = [agg(v['probs'], method) for v in data]

    best_f1_result = None
    best_f1_val = 0
    best_r90_result = None
    best_r90_precision = 0

    for t in np.arange(0.05, 0.96, 0.01):
        t = round(t, 2)
        TP, FP, FN, TN, p, r, f1, acc = metrics(scores, labels, t)
        if f1 > best_f1_val:
            best_f1_val = f1
            best_f1_result = (t, TP, FP, FN, TN, p, r, f1, acc)
        if r >= 0.90 and p > best_r90_precision:
            best_r90_precision = p
            best_r90_result = (t, TP, FP, FN, TN, p, r, f1, acc)

    print(f"\n[{method.upper()}]")
    if best_f1_result:
        t, TP, FP, FN, TN, p, r, f1, acc = best_f1_result
        print(f"  Best F1:      thresh={t:.2f}  F1={f1:.3f}  Recall={r:.3f}  Precision={p:.3f}  Acc={acc:.3f}  TP={TP} FP={FP} FN={FN} TN={TN}")
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_config = (method, best_f1_result)
    if best_r90_result:
        t, TP, FP, FN, TN, p, r, f1, acc = best_r90_result
        print(f"  Recall>=90%:  thresh={t:.2f}  F1={f1:.3f}  Recall={r:.3f}  Precision={p:.3f}  Acc={acc:.3f}  TP={TP} FP={FP} FN={FN} TN={TN}")
        if p > best_medical_precision:
            best_medical_precision = p
            best_medical_config = (method, best_r90_result)
    else:
        print(f"  Recall>=90%:  NOT ACHIEVABLE with this method")

print()
print('=' * 80)
print('OPTIMAL CONFIGURATIONS')
print('=' * 80)

if best_overall_config:
    method, (t, TP, FP, FN, TN, p, r, f1, acc) = best_overall_config
    print(f"\nBest F1 Config:")
    print(f"  Method: {method}, Threshold: {t:.2f}")
    print(f"  F1={f1:.3f} ({f1*100:.1f}%), Recall={r:.3f} ({r*100:.1f}%), Precision={p:.3f} ({p*100:.1f}%)")
    print(f"  Accuracy={acc:.3f} ({acc*100:.1f}%)")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")

if best_medical_config:
    method, (t, TP, FP, FN, TN, p, r, f1, acc) = best_medical_config
    print(f"\nBest Medical Config (Recall >= 90%):")
    print(f"  Method: {method}, Threshold: {t:.2f}")
    print(f"  F1={f1:.3f} ({f1*100:.1f}%), Recall={r:.3f} ({r*100:.1f}%), Precision={p:.3f} ({p*100:.1f}%)")
    print(f"  Accuracy={acc:.3f} ({acc*100:.1f}%)")
    print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")

# Per-patient analysis
print()
print('=' * 80)
print('PER-PATIENT ANALYSIS (max aggregation)')
print('=' * 80)

from collections import defaultdict
patient_data = defaultdict(lambda: {'seizure': [], 'normal': []})
for v in data:
    pid = v['patient_id']
    max_p = agg(v['probs'], 'max')
    if v['label'] == 1:
        patient_data[pid]['seizure'].append((v['name'], max_p))
    else:
        patient_data[pid]['normal'].append((v['name'], max_p))

print(f"\n{'Patient':<10} {'Seizure (avg max)':>18} {'Normal (avg max)':>16} {'Separation':>12}")
print("-" * 60)
for pid in sorted(patient_data.keys()):
    pd = patient_data[pid]
    s_probs = [x[1] for x in pd['seizure']]
    n_probs = [x[1] for x in pd['normal']]
    s_avg = np.mean(s_probs) if s_probs else None
    n_avg = np.mean(n_probs) if n_probs else None
    s_str = f"{s_avg:.3f}" if s_avg is not None else "  N/A"
    n_str = f"{n_avg:.3f}" if n_avg is not None else "  N/A"
    if s_avg is not None and n_avg is not None:
        sep = s_avg - n_avg
        sep_str = f"{sep:+.3f} {'GOOD' if sep > 0.3 else 'HARD' if abs(sep) < 0.15 else 'OK'}"
    else:
        sep_str = "  N/A"
    print(f"{pid:<10} {s_str:>18} {n_str:>16} {sep_str:>12}")

# Missed seizures analysis
print()
print('=' * 80)
print('MISSED SEIZURES ANALYSIS (at threshold=0.24, max aggregation)')
print('=' * 80)
missed = [v for v in data if v['label'] == 1 and agg(v['probs'], 'max') < 0.24]
print(f"\nMissed: {len(missed)}/42 seizure videos")
for v in missed:
    max_p = agg(v['probs'], 'max')
    n = v['n_windows']
    reason = "NO PERSON DETECTED" if n == 0 else f"Low confidence (max={max_p:.3f})"
    print(f"  {v['name']}: {reason} ({n} windows)")

# False alarms analysis
print()
print('FALSE ALARMS ANALYSIS (at threshold=0.24, max aggregation)')
print('=' * 80)
false_alarms = [v for v in data if v['label'] == 0 and agg(v['probs'], 'max') >= 0.24]
print(f"\nFalse Alarms: {len(false_alarms)}/42 normal videos")
for v in false_alarms:
    max_p = agg(v['probs'], 'max')
    print(f"  {v['name']}: MaxProb={max_p:.3f} (Patient {v['patient_id']})")
