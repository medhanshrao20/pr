"""Check saved classification evidence, temporal isolation and source integrity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

root = Path(__file__).resolve().parents[1]
out = root / 'results/classification'
meta = json.loads((out / 'metadata.json').read_text())
assign = pd.read_csv(out / 'split_assignments.csv')
predictions = pd.read_csv(out / 'test_predictions.csv')
scores = pd.read_csv(out / 'test_metrics.csv').set_index('model')
assert meta['instances'] >= 1000 and meta['raw_attributes'] >= 10
assert len(assign) == meta['instances'] and set(assign['class']) == {0, 1, 2}
assert assign.loc[assign.split.eq('train'), 'target_timestamp'].max() < assign.loc[assign.split.eq('test'), 'timestamp'].min()
assert assign.groupby('day').cv_fold.nunique().max() == 1
for fold in meta['folds']:
    assert not set(fold['train_days']) & set(fold['validation_days'])
    assert all(n > 0 for n in fold['train_class_counts'] + fold['validation_class_counts'])
for name, rows in predictions.groupby('model'):
    assert len(rows) == meta['test_n'] and not rows.timestamp.duplicated().any()
    assert np.isclose(accuracy_score(rows.truth, rows.prediction), scores.loc[name, 'accuracy'])
    assert np.isclose(f1_score(rows.truth, rows.prediction, average='macro'), scores.loc[name, 'macro_f1'])
    cm = pd.read_csv(out / ('confusion_' + name.lower().replace(' ', '_') + '.csv'), index_col=0)
    assert np.array_equal(cm.to_numpy(), confusion_matrix(rows.truth, rows.prediction))
folds = pd.read_csv(out / 'fold_scores.csv')
summary = pd.read_csv(out / 'cv_summary.csv').set_index('model')
for name, rows in folds.groupby('model'):
    assert len(rows) == 5
    for metric in ['accuracy', 'macro_f1']:
        assert np.isclose(rows[metric].mean(), summary.loc[name, metric + '_mean'])
        assert np.isclose(rows[metric].std(ddof=1), summary.loc[name, metric + '_std'])
manifest = json.loads((root / 'source_integrity.json').read_text())
checked = 0
for relative, expected in manifest.items():
    path = root / 'reference_forecasting' / relative
    if path.is_file():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
        checked += 1
print(f'PASS: metrics recompute, disjoint day groups, future holdout, class counts, CV standard deviations, {checked} preserved source files.')
