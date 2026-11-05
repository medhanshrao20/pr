"""Input alignment, geographic graphs, reproducible outputs, and evaluation."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ["wind_speed_mph", "air_temp_f", "relative_humidity_pct", "eto_in",
             "precip_in", "sol_rad_ly_per_day", "vapor_pressure_mbar", "dew_point_f", "soil_temp_f"]
NAMES = ["wind_ms", "air_temp_c", "relative_humidity_pct", "eto_mm", "precip_mm",
         "sol_rad_ly_per_day", "vapor_pressure_mbar", "dew_point_c", "soil_temp_c"]


def parser(stage, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--metadata", type=Path, default=ROOT / "data/station_coordinates.json")
    p.add_argument("--stations", nargs="+", type=int, help="Default: every station file (14).")
    p.add_argument("--exclude-stations", nargs="*", type=int, default=[])
    p.add_argument("--start", help="Inclusive timestamp; naive timestamps mean fixed PST.")
    p.add_argument("--end", help="Inclusive timestamp; naive timestamps mean fixed PST.")
    p.add_argument("--train-fraction", type=float, default=.7)
    p.add_argument("--val-fraction", type=float, default=.15)
    p.add_argument("--scope", choices=["train", "all"], default="train",
                   help="Analysis stages only; forecast graph estimation always uses train.")
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neighbors", type=int, default=3)
    p.add_argument("--qc", choices=["retain", "blank-nonempty"], default="retain",
                   help="Retain QC flags for audit, or conservatively mask any nonempty QC.")
    p.set_defaults(stage=stage)
    return p


def save_json(path, data):
    def convert(x):
        if isinstance(x, (Path, pd.Timestamp)): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False), encoding="utf-8")


def output_dir(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output or ROOT / args.stage / "results" / stamp
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "arguments.json", vars(args))
    save_json(out / "environment.json", {"python": platform.python_version(),
              "numpy": np.__version__, "pandas": pd.__version__})
    return out


def pst_bound(value):
    t = pd.Timestamp(value)
    return t.tz_localize("Etc/GMT+8") if t.tzinfo is None else t.tz_convert("Etc/GMT+8")


def load_panel(args, out):
    files = sorted(args.data.glob("station_*.csv"))
    if not files: raise ValueError(f"No station_*.csv files in {args.data}")
    frames, records, qc_counts = {}, [], []
    for path in files:
        df = pd.read_csv(path)
        ids = df.stn_id.dropna().unique()
        if len(ids) != 1: raise ValueError(f"Expected one station ID in {path}")
        sid = int(ids[0])
        if args.stations and sid not in args.stations: continue
        if sid in args.exclude_stations: continue
        if sid in frames: raise ValueError(f"Duplicate station {sid}")
        df.index = pd.to_datetime(df.timestamp_pst, utc=True).dt.tz_convert("Etc/GMT+8")
        if df.index.duplicated().any(): raise ValueError(f"Duplicate timestamps: {path}")
        df = df.sort_index()
        required = VARIABLES + ["wind_direction_deg"]
        if not set(required).issubset(df): raise ValueError(f"Missing measurements: {path}")
        values = df[required].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for col in required:
            # Explicit source-column mapping avoids ambiguous unit suffixes.
            qc = dict(zip(required, ["wind_speed_qc", "air_temp_qc", "relative_humidity_qc", "eto_qc",
                "precip_qc", "sol_rad_qc", "vapor_pressure_qc", "dew_point_qc", "soil_temp_qc", "wind_direction_qc"]))[col]
            flags = df[qc].fillna("").astype(str).str.strip() if qc in df else pd.Series("", index=df.index)
            qc_counts.append({"station": sid, "variable": col, "nonempty_qc": int(flags.ne("").sum())})
            if args.qc == "blank-nonempty": values.loc[flags.ne(""), col] = np.nan
        values.loc[values.wind_speed_mph < 0, "wind_speed_mph"] = np.nan
        values.loc[~values.relative_humidity_pct.between(0, 100), "relative_humidity_pct"] = np.nan
        values.loc[~values.wind_direction_deg.between(0, 360), "wind_direction_deg"] = np.nan
        values["wind_speed_mph"] *= .44704
        for c in ["air_temp_f", "dew_point_f", "soil_temp_f"]: values[c] = (values[c] - 32) * 5 / 9
        values[["eto_in", "precip_in"]] *= 25.4
        values.columns = NAMES + ["wind_direction_deg"]
        frames[sid] = values
        records.append({"station": sid, "name": str(df.stn_name.iloc[0]), "rows": len(df),
            "start": str(df.index.min()), "end": str(df.index.max()), "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if args.stations and set(args.stations) - set(frames) - set(args.exclude_stations):
        raise ValueError("A requested station was not found.")
    stations = sorted(frames)
    if len(stations) < 3: raise ValueError("At least three stations are required.")
    begin = max(f.index.min() for f in frames.values())
    end = min(f.index.max() for f in frames.values())
    if args.start: begin = max(begin, pst_bound(args.start))
    if args.end: end = min(end, pst_bound(args.end))
    times = pd.date_range(begin, end, freq="h")
    if len(times) < 100: raise ValueError("Fewer than 100 overlapping hourly timestamps.")
    raw = np.stack([frames[s].reindex(times).to_numpy(float) for s in stations], axis=1)
    if not 0 < args.train_fraction < 1 or not 0 < args.val_fraction < 1 or args.train_fraction + args.val_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than one.")
    train_end = int(len(times) * args.train_fraction)
    val_end = int(len(times) * (args.train_fraction + args.val_fraction))
    meta = pd.DataFrame(json.loads(args.metadata.read_text(encoding="utf-8"))).set_index("id")
    if meta.index.duplicated().any() or not set(stations).issubset(meta.index):
        raise ValueError("Missing or duplicate station coordinate metadata.")
    meta = meta.loc[stations]
    if not np.isfinite(meta[["lat", "lon"]].to_numpy(float)).all(): raise ValueError("Nonfinite coordinates.")
    if not meta.lat.between(-90, 90).all() or not meta.lon.between(-180, 180).all(): raise ValueError("Invalid coordinates.")
    meta.to_csv(out / "stations.csv", index_label="station")
    pd.DataFrame(records).to_csv(out / "input_audit.csv", index=False)
    pd.DataFrame(qc_counts).to_csv(out / "qc_audit.csv", index=False)
    save_json(out / "alignment.json", {"stations": stations, "hours": len(times), "start": str(times[0]),
        "end": str(times[-1]), "train_end_exclusive": str(times[train_end]),
        "validation_end_exclusive": str(times[val_end]), "variables": NAMES + ["wind_direction_deg"],
        "scope": args.scope, "missing_by_station_variable": np.isnan(raw).sum(axis=0),
        "coordinate_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "qc_policy": args.qc, "timezone": "fixed PST UTC-08, no DST"})
    return raw, times, stations, meta, train_end, val_end


def analysis_frame(raw, times, stations, train_end, scope):
    end = train_end if scope == "train" else len(times)
    return pd.DataFrame(raw[:end, :, 0], index=times[:end], columns=stations)


def anomalies(wind):
    # Remove each station's month/hour climatology, estimated only on the supplied scope.
    return wind - wind.groupby([wind.index.month, wind.index.hour]).transform("mean")


def geographic_distances(meta):
    lat, lon = np.radians(meta.lat.to_numpy()), np.radians(meta.lon.to_numpy())
    a = np.sin((lat[:, None]-lat)/2)**2 + np.cos(lat[:, None])*np.cos(lat)*np.sin((lon[:, None]-lon)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_weights(distance, neighbors):
    n = len(distance)
    if not 1 <= neighbors < n: raise ValueError("neighbors must be between 1 and N-1.")
    d = distance.copy()
    np.fill_diagonal(d, np.inf)
    w = np.zeros_like(d)
    for i in range(n):
        j = np.argsort(d[i], kind="stable")[:neighbors]
        w[i, j] = 1
    return np.maximum(w, w.T)


def row_normalize(w):
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)


def build_graph(kind, meta, train_wind, neighbors):
    d = geographic_distances(meta)
    support = knn_weights(d, neighbors)
    scale = max(float(np.median(d[support > 0])), 1e-6)
    geo = support * np.exp(-(d/scale)**2)
    if kind == "geographic": a = geo
    else:
        corr = train_wind.corr(min_periods=48).to_numpy()
        if not np.isfinite(corr).all(): raise ValueError("Insufficient/nonvarying training wind for graph correlations.")
        distance = 1 - np.clip(corr, -1, 1)
        stat = knn_weights(distance, neighbors) * np.maximum(corr, 0)
        a = stat if kind == "statistical" else .5*(geo + stat)
    # Retain an isolated station through its self loop; report all saved graph weights.
    np.fill_diagonal(a, 1)
    return row_normalize(a).astype(np.float32)


def heatmap(matrix, labels, path, title, signed=False):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="coolwarm" if signed else "viridis",
                   vmin=-1 if signed else None, vmax=1 if signed else None)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def season_labels(times):
    return np.array(["winter" if m in (12, 1, 2) else "spring" if m in (3, 4, 5)
                     else "summer" if m in (6, 7, 8) else "autumn" for m in times.month])


def metric_rows(actual, predicted, stations, seasons, model):
    if actual.shape != predicted.shape:
        raise ValueError("Prediction and target shapes differ.")
    if np.any(np.isfinite(actual) & ~np.isfinite(predicted)):
        raise FloatingPointError("Nonfinite prediction at an observed target; refusing to silently exclude it.")
    rows = []
    for n, sid in enumerate(stations):
        for h in range(actual.shape[-1]):
            for season in ["all", "spring", "summer", "autumn", "winter"]:
                mask = np.isfinite(actual[:, n, h]) & np.isfinite(predicted[:, n, h])
                if season != "all": mask &= seasons[:, h] == season
                y, p = actual[mask, n, h], predicted[mask, n, h]
                if not len(y): continue
                e = p-y
                eligible = np.abs(y) >= .1
                rows.append({"model": model, "station": sid, "season": season, "horizon": h+1,
                    "n": len(y), "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))),
                    "mape_pct": float(np.mean(np.abs(e[eligible]/y[eligible]))*100) if eligible.any() else None,
                    "mape_n": int(eligible.sum()), "mape_min_abs_target_ms": .1})
    return rows
