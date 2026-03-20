import json
with open('seizure_detection/report_eval/full_pipeline_results.json') as f:
    d = json.load(f)
m = d['metrics']
print('=== FULL PIPELINE RESULTS ===')
print(f"Accuracy:  {m['accuracy']:.1%}")
print(f"Precision: {m['precision']:.1%}")
print(f"Recall:    {m['recall']:.1%}")
print(f"F1-Score:  {m['f1']:.1%}")
print(f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")
print()
videos = d['per_video']
print(f'Total videos evaluated: {len(videos)}')
missed = [v for v in videos if v['label']==1 and not v['detected']]
false_alarms = [v for v in videos if v['label']==0 and v['detected']]
print(f'Missed seizures: {len(missed)}')
for v in missed:
    print(f"  {v['name']}: MaxProb={v['max_prob']:.3f} RhythmSuppressed={v['rhythm_suppressed']}")
print(f'False alarms: {len(false_alarms)}')
for v in false_alarms:
    print(f"  {v['name']}: MaxProb={v['max_prob']:.3f} RhythmFires={v['rhythm_fires']} Patient={v['patient_id']}")
total_suppressed = sum(v['rhythm_suppressed'] for v in videos)
total_rhythm_fires = sum(v['rhythm_fires'] for v in videos)
print(f'Total rhythm confirmations: {total_rhythm_fires}')
print(f'Total rhythm suppressions: {total_suppressed}')
