"""Verify saved metrics, matched folds, source preservation and causal features."""
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from run_experiment import ROOT, build_dataset, past_features


def main():
    out = ROOT / 'results/classification'
    info = json.loads((out / 'metadata.json').read_text())
    assignments = pd.read_csv(out / 'split_assignments.csv')
    predictions = pd.read_csv(out / 'test_predictions.csv')
    cv_predictions = pd.read_csv(out / 'cv_predictions.csv')
    test = pd.read_csv(out / 'test_metrics.csv').set_index(['condition', 'model'])
    folds = pd.read_csv(out / 'fold_scores.csv')
    summary = pd.read_csv(out / 'cv_summary.csv').set_index(['condition', 'model'])
    assert info['instances'] >= 1000 and info['raw_attributes'] >= 10
    assert len(assignments) == info['instances'] and set(assignments['class']) == {0, 1, 2}
    assert assignments.groupby('day').cv_fold.nunique().max() == 1
    assert assignments.query('split == "train"').target_timestamp.max() < assignments.query('split == "test"').timestamp.min()
    ref_rows = None
    for (condition, model), rows in predictions.groupby(['condition', 'model']):
        assert len(rows) == info['test_n'] and not rows.timestamp.duplicated().any()
        matched = rows[['timestamp', 'target_timestamp', 'truth']].reset_index(drop=True)
        if ref_rows is None: ref_rows = matched
        pd.testing.assert_frame_equal(matched, ref_rows)
        assert np.isclose(accuracy_score(rows.truth, rows.prediction), test.loc[(condition, model), 'accuracy'])
        assert np.isclose(f1_score(rows.truth, rows.prediction, average='macro'), test.loc[(condition, model), 'macro_f1'])
        saved = pd.read_csv(out / f'confusion_{condition}_{model}.csv', index_col=0).to_numpy()
        assert np.array_equal(saved, confusion_matrix(rows.truth, rows.prediction, labels=[0, 1, 2]))
    for (condition, model), rows in cv_predictions.groupby(['condition', 'model']):
        assert len(rows) == info['train_n'] and not rows.row.duplicated().any()
        for fold, part in rows.groupby('fold'):
            assert set(part.row) == set(assignments.index[assignments.cv_fold.eq(fold)])
            score = folds.query('condition == @condition and model == @model and fold == @fold').iloc[0]
            assert np.isclose(accuracy_score(part.truth, part.prediction), score.accuracy)
            assert np.isclose(f1_score(part.truth, part.prediction, average='macro'), score.macro_f1)
    for key, rows in folds.groupby(['condition', 'model']):
        assert len(rows) == 5
        for metric in ['accuracy', 'macro_f1']:
            assert np.isclose(rows[metric].mean(), summary.loc[key, metric + '_mean'])
            assert np.isclose(rows[metric].std(ddof=1), summary.loc[key, metric + '_std'])
    source = json.loads((ROOT / 'docs/original_source_manifest.json').read_text())
    for name, expected in source['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    # Changing later data must never change features computed at earlier times.
    times = pd.date_range('2024-01-01', periods=50, freq='h', tz='Etc/GMT+8')
    frame = pd.DataFrame({'wind_speed_mph': np.arange(50, dtype=float)}, index=times)
    before = past_features(frame)
    frame.loc[times[12]:, 'wind_speed_mph'] = 9999
    after = past_features(frame)
    pd.testing.assert_frame_equal(before.iloc[:12], after.iloc[:12])
    assert pd.isna(before.iloc[24].wind_lag_1), 'History must reset at a day boundary'
    # Recompute dataset and fixed-neighbor choices, not model fitting.
    features, y, meta, neighbors, _ = build_dataset()
    assert neighbors == info['neighbors']
    for key, x in features.items():
        assert x.shape[1] == info['feature_counts'][key]
    assert np.array_equal(y, assignments['class'])
    paired = pd.read_csv(out / 'paired_comparison.csv')
    for _, r in paired.iterrows():
        for metric in ['accuracy', 'macro_f1']:
            actual = test.loc[('spatiotemporal', r.model), metric] - test.loc[('temporal', r.model), metric]
            assert np.isclose(actual, r['test_' + metric + '_difference'])
    print(f'PASS: recomputed metrics, matched fold/test rows, temporal isolation, causal histories and all {len(source["files"])} original hashes.')


if __name__ == '__main__':
    main()
