"""Classical next-hour wind-regime classification; original forecasting code is untouched."""
import hashlib
import json
import platform
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
OUT = ROOT / 'results' / 'classification'
RAW = ['eto_in', 'precip_in', 'sol_rad_ly_per_day', 'vapor_pressure_mbar',
       'air_temp_f', 'relative_humidity_pct', 'dew_point_f', 'wind_speed_mph',
       'wind_direction_deg', 'soil_temp_f']
CLASSES = ['Low (<3 mph)', 'Moderate (3 to <7 mph)', 'High (>=7 mph)']


def load_data():
    path = ROOT / 'reference_forecasting/data/station_047_brentwood.csv'
    df = pd.read_csv(path)
    ts = pd.to_datetime(df.timestamp_pst, utc=True).dt.tz_convert('Etc/GMT+8')
    assert ts.is_unique and ts.is_monotonic_increasing
    day = ts.dt.strftime('%Y-%m-%d')
    x = df[RAW].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    future = x.wind_speed_mph.shift(-1)
    valid = ((ts.shift(-1) - ts == pd.Timedelta(hours=1)) & day.eq(day.shift(-1))
             & future.notna() & future.ge(0) & ts.dt.year.isin([2024, 2025]))
    y = pd.Series(np.digitize(future, [3., 7.]), index=x.index)
    enhanced = x.copy()
    grouped = x.groupby(day, sort=False)
    enhanced['wind_lag_1'] = grouped.wind_speed_mph.shift(1)
    enhanced['wind_lag_3'] = grouped.wind_speed_mph.shift(3)
    enhanced['wind_mean_3'] = grouped.wind_speed_mph.transform(lambda s: s.rolling(3, min_periods=1).mean())
    enhanced['wind_std_3'] = grouped.wind_speed_mph.transform(lambda s: s.rolling(3, min_periods=2).std())
    enhanced['wind_change'] = x.wind_speed_mph - enhanced.wind_lag_1
    enhanced['temperature_dewpoint_spread'] = x.air_temp_f - x.dew_point_f
    for name, angle in [('direction', x.wind_direction_deg * np.pi / 180),
                        ('hour', ts.dt.hour * 2 * np.pi / 24),
                        ('annual', (ts.dt.dayofyear - 1) * 2 * np.pi / 365.25)]:
        enhanced[name + '_sin'] = np.sin(angle)
        enhanced[name + '_cos'] = np.cos(angle)
    meta = pd.DataFrame({'timestamp': ts.astype(str), 'target_timestamp': ts.shift(-1).astype(str),
                         'day': day, 'year': ts.dt.year})
    return x[valid].reset_index(drop=True), enhanced[valid].reset_index(drop=True), y[valid].reset_index(drop=True), meta[valid].reset_index(drop=True), path


def metrics(y, pred):
    return {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, average='macro', zero_division=0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, enhanced, y, meta, source = load_data()
    train = np.flatnonzero(meta.year.eq(2024))
    test = np.flatnonzero(meta.year.eq(2025))
    assert len(x) >= 1000 and x.shape[1] >= 10 and set(y) == {0, 1, 2}
    assert meta.loc[train, 'target_timestamp'].max() < meta.loc[test, 'timestamp'].min()
    models = {
        'Logistic raw': (x, LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42)),
        'Logistic features': (enhanced, LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42)),
        'LDA features': (enhanced, LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        'Random forest features': (enhanced, RandomForestClassifier(n_estimators=160, max_depth=14,
            min_samples_leaf=3, class_weight='balanced_subsample', n_jobs=2, random_state=42)),
    }
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(x.iloc[train], y.iloc[train], groups=meta.day.iloc[train]))
    records, class_records, final_records, predictions, audits = [], [], [], [], []
    for fold, (a, b) in enumerate(folds, 1):
        fit, val = train[a], train[b]
        assert not set(meta.day.iloc[fit]) & set(meta.day.iloc[val])
        audits.append({'fold': fold, 'train_n': len(fit), 'validation_n': len(val),
                       'train_class_counts': np.bincount(y.iloc[fit], minlength=3).tolist(),
                       'validation_class_counts': np.bincount(y.iloc[val], minlength=3).tolist(),
                       'train_days': sorted(set(meta.day.iloc[fit])), 'validation_days': sorted(set(meta.day.iloc[val]))})
        for name, (features, model) in models.items():
            pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clone(model))
            pipe.fit(features.iloc[fit], y.iloc[fit])
            pred = pipe.predict(features.iloc[val])
            records.append({'model': name, 'fold': fold, **metrics(y.iloc[val], pred)})
        records.append({'model': 'Persistence', 'fold': fold,
                        **metrics(y.iloc[val], np.digitize(x.wind_speed_mph.iloc[val].fillna(x.wind_speed_mph.iloc[fit].median()), [3., 7.]))})
        print(f'Completed validation fold {fold}/5', flush=True)
    fold_scores = pd.DataFrame(records)
    summary = fold_scores.groupby('model')[['accuracy', 'macro_f1']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    selected = summary.loc[list(models), 'macro_f1_mean'].idxmax()
    for name in [*models, 'Persistence']:
        if name == 'Persistence':
            pred = np.digitize(x.wind_speed_mph.iloc[test].fillna(x.wind_speed_mph.iloc[train].median()), [3., 7.])
        else:
            features, model = models[name]
            pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), clone(model))
            pipe.fit(features.iloc[train], y.iloc[train])
            pred = pipe.predict(features.iloc[test])
        final_records.append({'model': name, **metrics(y.iloc[test], pred)})
        report = classification_report(y.iloc[test], pred, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0)
        for label in CLASSES:
            class_records.append({'model': name, 'class': label, **report[label]})
        slug = name.lower().replace(' ', '_')
        cm = confusion_matrix(y.iloc[test], pred, labels=[0, 1, 2])
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(OUT / f'confusion_{slug}.csv')
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        ConfusionMatrixDisplay(cm, display_labels=['Low', 'Moderate', 'High']).plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
        ax.set_title(name + ': held-out 2025')
        fig.tight_layout(); fig.savefig(OUT / f'confusion_{slug}.png', dpi=170); plt.close(fig)
        predictions.append(pd.DataFrame({'model': name, 'timestamp': meta.timestamp.iloc[test].to_numpy(),
            'target_timestamp': meta.target_timestamp.iloc[test].to_numpy(), 'truth': y.iloc[test].to_numpy(), 'prediction': pred}))
    fold_scores.to_csv(OUT / 'fold_scores.csv', index=False)
    summary.to_csv(OUT / 'cv_summary.csv')
    final = pd.DataFrame(final_records)
    final.to_csv(OUT / 'test_metrics.csv', index=False)
    pd.DataFrame(class_records).to_csv(OUT / 'per_class_test.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'test_predictions.csv', index=False)
    assignment = meta.copy(); assignment['split'] = np.where(meta.year.eq(2024), 'train', 'test'); assignment['class'] = y
    assignment['cv_fold'] = 0
    for fold, (_, b) in enumerate(folds, 1): assignment.loc[train[b], 'cv_fold'] = fold
    assignment.to_csv(OUT / 'split_assignments.csv', index=False)
    metadata = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'raw_rows': len(pd.read_csv(source)),
        'instances': len(y), 'raw_attributes': len(RAW), 'engineered_attributes': enhanced.shape[1],
        'train_n': len(train), 'test_n': len(test), 'train_counts': np.bincount(y.iloc[train], minlength=3).tolist(),
        'test_counts': np.bincount(y.iloc[test], minlength=3).tolist(), 'class_names': CLASSES,
        'raw_features': RAW, 'engineered_features': enhanced.columns.tolist(), 'selected_model': selected,
        'selection': 'Highest 2024 grouped cross-validation mean macro-F1 among learned models',
        'classes': 'Fixed project thresholds for next-hour wind speed in mph: <3, [3,7), >=7. Not an official warning scale.',
        'protocol': '2024 training, 5-fold stratified day-group cross-validation, untouched 2025 final test. Same-day histories and targets only.',
        'sd': 'Sample standard deviation across five validation folds (ddof=1), not uncertainty over independent datasets.',
        'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
        'seed': 42, 'folds': audits}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\n2024 cross-validation mean and sample standard deviation:\n' + summary.to_string())
    print('\nSelected before final test: ' + selected)
    print('\n2025 test results:\n' + final.to_string(index=False))
    print('\n2025 per-class precision and recall:\n' + pd.DataFrame(class_records).to_string(index=False))
    print('\nSelected model confusion matrix (rows=true, columns=predicted):\n' + pd.read_csv(OUT / ('confusion_' + selected.lower().replace(' ', '_') + '.csv'), index_col=0).to_string())
    print('\nAll tables, predictions, fold assignments and confusion figures: ' + str(OUT), flush=True)


if __name__ == '__main__':
    main()
