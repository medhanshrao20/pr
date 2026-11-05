"""Matched classical classification study of temporal and spatial information."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ['eto_in', 'precip_in', 'sol_rad_ly_per_day', 'vapor_pressure_mbar', 'air_temp_f',
       'relative_humidity_pct', 'dew_point_f', 'wind_speed_mph', 'wind_direction_deg', 'soil_temp_f']
CLASSES = ['Low (<3 mph)', 'Medium (3 to <7 mph)', 'High (>=7 mph)']
CONDITIONS = ['local_current', 'temporal', 'spatial_current', 'spatiotemporal']
SEED = 42


def read_station(path):
    raw = pd.read_csv(path)
    timestamps = pd.DatetimeIndex(pd.to_datetime(raw.timestamp_pst, utc=True).dt.tz_convert('Etc/GMT+8'))
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError(f'Duplicate or unsorted timestamps in {path}')
    if raw.stn_id.nunique() != 1:
        raise ValueError(f'Multiple station IDs in {path}')
    frame = raw[RAW].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    frame.index = timestamps
    return int(raw.stn_id.iloc[0]), frame


def past_features(frame):
    day = frame.index.strftime('%Y-%m-%d')
    wind = frame.wind_speed_mph
    groups = wind.groupby(day)
    result = pd.DataFrame(index=frame.index)
    result['wind_lag_1'] = groups.shift(1)
    result['wind_lag_3'] = groups.shift(3)
    result['wind_mean_3'] = groups.transform(lambda v: v.rolling(3, min_periods=1).mean())
    result['wind_std_3'] = groups.transform(lambda v: v.rolling(3, min_periods=2).std())
    result['wind_change'] = wind - result.wind_lag_1
    return result


def build_dataset(data=ROOT / 'data'):
    coordinates = pd.DataFrame(json.loads((data / 'station_coordinates.json').read_text())).set_index('id')
    if coordinates.index.duplicated().any():
        raise ValueError('Duplicate station coordinates')
    stations, audits = {}, []
    for path in sorted(data.glob('station_*.csv')):
        sid, frame = read_station(path)
        if sid in stations or sid not in coordinates.index:
            raise ValueError('Duplicate station or missing coordinates')
        stations[sid] = frame
        audits.append({'station': sid, 'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'rows': len(frame), 'missing_predictor_cells': int(frame.isna().sum().sum())})
    target = stations[47]
    # A complete hourly index makes a one-row shift an actual one-hour lag.
    target = target.reindex(pd.date_range(target.index.min(), target.index.max(), freq='h'))
    lat0, lon0 = np.radians(coordinates.loc[47, ['lat', 'lon']].to_numpy(float))
    distances = []
    for sid in stations:
        if sid == 47:
            continue
        lat, lon = np.radians(coordinates.loc[sid, ['lat', 'lon']].to_numpy(float))
        a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
        km = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        distances.append({'station': sid, 'name': str(coordinates.loc[sid, 'name']), 'distance_km': float(km)})
    neighbors = sorted(distances, key=lambda r: (r['distance_km'], r['station']))[:3]
    if len(neighbors) != 3:
        raise ValueError('Three neighboring station files are required')
    weights = np.array([1 / max(r['distance_km'], 1e-6) for r in neighbors])
    weights /= weights.sum()
    for row, weight in zip(neighbors, weights):
        row['inverse_distance_weight'] = float(weight)
    times = target.index
    day = pd.Series(times.strftime('%Y-%m-%d'), index=times)
    future = target.wind_speed_mph.shift(-1)
    valid = day.eq(day.shift(-1)) & future.notna() & future.ge(0) & np.isin(times.year, [2024, 2025])
    labels = pd.Series(np.digitize(future, [3., 7.]), index=times)
    local = target.copy()
    temporal = pd.concat([local, past_features(target)], axis=1)
    temporal['temperature_dewpoint_spread'] = local.air_temp_f - local.dew_point_f
    for name, angle in [('direction', local.wind_direction_deg * np.pi / 180),
                        ('hour', times.hour * 2 * np.pi / 24),
                        ('annual', (times.dayofyear - 1) * 2 * np.pi / 365.25)]:
        temporal[name + '_sin'] = np.sin(angle)
        temporal[name + '_cos'] = np.cos(angle)
    current_neighbors, history_neighbors, aligned = [], [], {}
    fields = ['wind_speed_mph', 'air_temp_f', 'relative_humidity_pct', 'vapor_pressure_mbar']
    for n in neighbors:
        sid = n['station']
        aligned[sid] = stations[sid].reindex(times)
        current_neighbors.append(aligned[sid][fields].add_prefix(f'n{sid}_'))
        history_neighbors.append(past_features(aligned[sid])[['wind_lag_1', 'wind_mean_3']].add_prefix(f'n{sid}_'))
        for year in [2024, 2025]:
            mask = (times.year == year) & valid.to_numpy()
            n[f'missing_neighbor_cells_{year}'] = int(aligned[sid].loc[mask, fields].isna().sum().sum())
    current_neighbors = pd.concat(current_neighbors, axis=1)
    history_neighbors = pd.concat(history_neighbors, axis=1)
    aggregate = pd.DataFrame(index=times)
    for col in ['wind_speed_mph', 'air_temp_f', 'relative_humidity_pct']:
        values = np.column_stack([aligned[n['station']][col] for n in neighbors])
        available = np.isfinite(values)
        denominator = (available * weights).sum(axis=1)
        mean = np.divide(np.nansum(values * weights, axis=1), denominator,
                         out=np.full(len(times), np.nan), where=denominator > 0)
        aggregate['neighbor_mean_' + col] = mean
        aggregate['local_minus_neighbor_' + col] = local[col] - mean
    features = {
        'local_current': local,
        'temporal': temporal,
        'spatial_current': pd.concat([local, current_neighbors, aggregate], axis=1),
        'spatiotemporal': pd.concat([temporal, current_neighbors, history_neighbors, aggregate], axis=1),
    }
    features = {k: v.loc[valid].reset_index(drop=True) for k, v in features.items()}
    meta = pd.DataFrame({'timestamp': times.astype(str), 'target_timestamp': pd.Series(times.astype(str), index=times).shift(-1),
                         'day': day, 'year': times.year}).loc[valid].reset_index(drop=True)
    return features, labels.loc[valid].reset_index(drop=True), meta, neighbors, audits


def scores(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0))}


def macro_from_cm(cm):
    tp = np.diag(cm)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    return np.divide(2 * tp, denominator, out=np.zeros(3), where=denominator > 0).mean()


def paired_day_bootstrap(y, pred_temporal, pred_spatial, days):
    unique = np.unique(days)
    a, b = [], []
    for day in unique:
        mask = np.asarray(days) == day
        a.append(confusion_matrix(np.asarray(y)[mask], np.asarray(pred_temporal)[mask], labels=[0, 1, 2]))
        b.append(confusion_matrix(np.asarray(y)[mask], np.asarray(pred_spatial)[mask], labels=[0, 1, 2]))
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(2000):
        draw = rng.integers(len(unique), size=len(unique))
        cm_a, cm_b = a[draw].sum(axis=0), b[draw].sum(axis=0)
        differences.append((np.trace(cm_b) / cm_b.sum() - np.trace(cm_a) / cm_a.sum(), macro_from_cm(cm_b) - macro_from_cm(cm_a)))
    interval = np.percentile(differences, [2.5, 97.5], axis=0)
    return {'accuracy_difference_ci_low': float(interval[0, 0]), 'accuracy_difference_ci_high': float(interval[1, 0]),
            'macro_f1_difference_ci_low': float(interval[0, 1]), 'macro_f1_difference_ci_high': float(interval[1, 1]),
            'bootstrap_replicates': 2000, 'bootstrap_unit': 'calendar day', 'test_days': len(unique)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/classification')
    args = parser.parse_args(argv)
    if len(set(args.conditions)) != len(args.conditions):
        parser.error('Conditions must be unique')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    features, y, meta, neighbors, audits = build_dataset()
    train = np.flatnonzero(meta.year.eq(2024)); test = np.flatnonzero(meta.year.eq(2025))
    assert len(y) >= 1000 and features['local_current'].shape[1] >= 10 and set(y) == {0, 1, 2}
    assert meta.target_timestamp.iloc[train].max() < meta.timestamp.iloc[test].min()
    models = {'Logistic': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
              'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
              'RandomForest': RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=3,
                   class_weight='balanced_subsample', random_state=SEED, n_jobs=2)}
    cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(features['temporal'].iloc[train], y.iloc[train], meta.day.iloc[train]))
    folds, assignments, cv_predictions = [], meta.copy(), []
    assignments['class'] = y; assignments['split'] = np.where(meta.year.eq(2024), 'train', 'test'); assignments['cv_fold'] = 0
    fold_audit = []
    for fold, (a, b) in enumerate(cv, 1):
        fit, val = train[a], train[b]
        assert not set(meta.day.iloc[fit]) & set(meta.day.iloc[val])
        assert set(y.iloc[fit]) == set(y.iloc[val]) == {0, 1, 2}
        assignments.loc[val, 'cv_fold'] = fold
        fold_audit.append({'fold': fold, 'train_n': len(fit), 'validation_n': len(val),
                           'train_counts': np.bincount(y.iloc[fit], minlength=3).tolist(),
                           'validation_counts': np.bincount(y.iloc[val], minlength=3).tolist()})
        for condition in args.conditions:
            x = features[condition]
            for name, model in models.items():
                start = time.perf_counter()
                pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clone(model))
                pipe.fit(x.iloc[fit], y.iloc[fit]); pred = pipe.predict(x.iloc[val])
                folds.append({'condition': condition, 'model': name, 'fold': fold, **scores(y.iloc[val], pred),
                              'elapsed_seconds': time.perf_counter() - start})
                cv_predictions.append(pd.DataFrame({'condition': condition, 'model': name, 'row': val,
                    'fold': fold, 'truth': y.iloc[val].to_numpy(), 'prediction': pred}))
            print(f'Validation fold {fold}/5: {condition}', flush=True)
        persistence = np.digitize(features['local_current'].wind_speed_mph.iloc[val].fillna(features['local_current'].wind_speed_mph.iloc[fit].median()), [3., 7.])
        folds.append({'condition': 'persistence', 'model': 'Persistence', 'fold': fold, **scores(y.iloc[val], persistence), 'elapsed_seconds': 0.})
    fold_scores = pd.DataFrame(folds)
    summary = fold_scores.groupby(['condition', 'model'])[['accuracy', 'macro_f1']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    # Fix the classifier using only the temporal arm's validation scores.
    primary_model = summary.loc['temporal', 'macro_f1_mean'].idxmax() if 'temporal' in args.conditions else None
    selection = {'primary_classifier': primary_model, 'selection_rule': 'Maximum temporal mean validation macro-F1, then compare the same classifier in both arms.'}
    (out / 'selection_before_test.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    final, per_class, predictions, prediction_map = [], [], [], {}
    tests = [(c, m, features[c], model) for c in args.conditions for m, model in models.items()]
    tests.append(('persistence', 'Persistence', features['local_current'], None))
    for condition, name, x, model in tests:
        if model is None:
            pred = np.digitize(x.wind_speed_mph.iloc[test].fillna(x.wind_speed_mph.iloc[train].median()), [3., 7.])
        else:
            pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clone(model))
            pipe.fit(x.iloc[train], y.iloc[train]); pred = pipe.predict(x.iloc[test])
        prediction_map[(condition, name)] = pred
        final.append({'condition': condition, 'model': name, 'features': x.shape[1], **scores(y.iloc[test], pred)})
        report = classification_report(y.iloc[test], pred, labels=[0, 1, 2], target_names=CLASSES, zero_division=0, output_dict=True)
        for label in CLASSES:
            per_class.append({'condition': condition, 'model': name, 'class': label, **report[label]})
        slug = condition + '_' + name
        cm = confusion_matrix(y.iloc[test], pred, labels=[0, 1, 2])
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / f'confusion_{slug}.csv')
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        ConfusionMatrixDisplay(cm, display_labels=['Low', 'Medium', 'High']).plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
        ax.set_title(condition + ' / ' + name + '\nHeld-out 2025')
        fig.tight_layout(); fig.savefig(out / f'confusion_{slug}.png', dpi=170); plt.close(fig)
        predictions.append(pd.DataFrame({'condition': condition, 'model': name, 'timestamp': meta.timestamp.iloc[test].to_numpy(),
            'target_timestamp': meta.target_timestamp.iloc[test].to_numpy(), 'day': meta.day.iloc[test].to_numpy(),
            'truth': y.iloc[test].to_numpy(), 'prediction': pred}))
        print(f'Final evaluation: {condition} / {name}', flush=True)
    paired = []
    if {'temporal', 'spatiotemporal'}.issubset(args.conditions):
        for name in models:
            a = fold_scores.query('condition == "temporal" and model == @name').set_index('fold')
            b = fold_scores.query('condition == "spatiotemporal" and model == @name').set_index('fold')
            p_a, p_b = prediction_map[('temporal', name)], prediction_map[('spatiotemporal', name)]
            test_a, test_b = scores(y.iloc[test], p_a), scores(y.iloc[test], p_b)
            paired.append({'model': name, 'primary': name == primary_model,
                'cv_accuracy_difference_mean': float((b.accuracy - a.accuracy).mean()),
                'cv_accuracy_difference_std': float((b.accuracy - a.accuracy).std(ddof=1)),
                'cv_macro_f1_difference_mean': float((b.macro_f1 - a.macro_f1).mean()),
                'cv_macro_f1_difference_std': float((b.macro_f1 - a.macro_f1).std(ddof=1)),
                'test_accuracy_difference': test_b['accuracy'] - test_a['accuracy'],
                'test_macro_f1_difference': test_b['macro_f1'] - test_a['macro_f1'],
                **paired_day_bootstrap(y.iloc[test], p_a, p_b, meta.day.iloc[test])})
    metadata = {'seed': SEED, 'target_station': 47, 'classes': CLASSES, 'thresholds_mph': [3., 7.],
        'raw_attributes': len(RAW), 'instances': len(y), 'train_n': len(train), 'test_n': len(test),
        'train_class_counts': np.bincount(y.iloc[train], minlength=3).tolist(),
        'test_class_counts': np.bincount(y.iloc[test], minlength=3).tolist(), 'neighbors': neighbors,
        'feature_names': {k: v.columns.tolist() for k, v in features.items()},
        'feature_counts': {k: v.shape[1] for k, v in features.items()}, 'executed_conditions': args.conditions,
        'protocol': '2024 development, five stratified calendar-day grouped folds; 2025 final future test; same-day histories and next-hour labels.',
        'hypothesis': 'Adding spatial current observations and histories to temporal features increases next-hour macro-F1 for the same classifier.',
        **selection, 'folds': fold_audit, 'input_audit': audits,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scikit_learn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
        'classifiers': {k: v.get_params() for k, v in models.items()}}
    fold_scores.to_csv(out / 'fold_scores.csv', index=False)
    summary.to_csv(out / 'cv_summary.csv')
    pd.DataFrame(final).to_csv(out / 'test_metrics.csv', index=False)
    pd.DataFrame(per_class).to_csv(out / 'per_class_test.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / 'test_predictions.csv', index=False)
    pd.concat(cv_predictions, ignore_index=True).to_csv(out / 'cv_predictions.csv', index=False)
    assignments.to_csv(out / 'split_assignments.csv', index=False)
    pd.DataFrame(paired).to_csv(out / 'paired_comparison.csv', index=False)
    pd.DataFrame(neighbors).to_csv(out / 'neighbors.csv', index=False)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(models):
        group = pd.DataFrame(final).query('model == @name').set_index('condition').loc[args.conditions]
        ax.bar(np.arange(len(group)) + (i - 1) * .24, group.macro_f1, width=.24, label=name)
    ax.set_xticks(np.arange(len(args.conditions)), [c.replace('_', '\n') for c in args.conditions]); ax.set_ylim(0, 1)
    ax.set_ylabel('Macro-F1'); ax.set_title('Held-out 2025: matched classifier comparisons'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'ablation_macro_f1.png', dpi=180); plt.close(fig)
    print('\n2024 validation mean and sample SD:\n' + summary.to_string())
    print('\n2025 final test:\n' + pd.DataFrame(final).to_string(index=False))
    print('\nPaired spatial minus temporal comparisons:\n' + pd.DataFrame(paired).to_string(index=False))
    print('\nAll per-class precision and recall:\n' + pd.DataFrame(per_class).to_string(index=False))
    if primary_model:
        for condition in ['temporal', 'spatiotemporal']:
            if condition in args.conditions:
                print(f'\n{condition} / {primary_model} confusion matrix (rows true):\n' + pd.read_csv(out / f'confusion_{condition}_{primary_model}.csv', index_col=0).to_string())
    print('\nResults: ' + str(out), flush=True)


if __name__ == '__main__':
    main()
