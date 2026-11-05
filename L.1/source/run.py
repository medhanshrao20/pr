from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
import unittest
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from decomposition import decompose, relative_reconstruction_error, vmd_modes
from model import KANInformer

PIPELINE_NAME = "original_dataleak.v1"
DECOMPOSE_BEFORE_SPLIT = True
ROOT = Path(__file__).resolve().parent
VARIABLES = ["ET", "PCP", "SR", "VP", "AT", "RH", "DPT", "WS", "WD", "ST"]
SEASONS = ["spring", "summer", "autumn", "winter"]
ALPHA_GRID = (25.0, 100.0, 500.0, 2000.0)
PERIODS = {"spring": ("2021-03-01", "2021-06-01"), "summer": ("2021-06-01", "2021-09-01"),
           "autumn": ("2021-09-01", "2021-12-01"), "winter": ("2020-12-01", "2021-03-01")}
PAPER_INPUTS = {"spring": ["WS", "ET"], "summer": ["WS", "ET", "AT", "RH"],
                "autumn": ["WS", "ET"], "winter": ["WS"]}
PAPER_K = {"spring": 11, "summer": 10, "autumn": 11, "winter": 10}
PAPER_METRICS = {
    "spring": [(0.071, 0.056, 2.0), (0.100, 0.078, 2.7), (0.112, 0.090, 3.3)],
    "summer": [(0.064, 0.051, 3.1), (0.086, 0.066, 3.9), (0.115, 0.091, 5.8)],
    "autumn": [(0.061, 0.048, 3.8), (0.078, 0.063, 5.2), (0.084, 0.066, 5.4)],
    "winter": [(0.119, 0.089, 4.5), (0.160, 0.125, 6.3), (0.192, 0.148, 7.8)],
}


@dataclass
class Config:
    architecture: str = "paper"
    epochs: int = 300
    batch_size: int = 32
    lr: float = 0.001
    d_model: int = 64
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    factor: int = 5
    dropout: float = 0.0
    window: int = 7
    horizons: int = 3
    label_len: int = 4
    kan_hidden: int = 20
    kan_grid: int = 5
    patience: int = 30
    seed: int = 42
    vmd_alpha: float = 100.0
    tune_alpha: bool = True
    vmd_k_min: int = 2
    vmd_k_max: int = 12
    vmd_tau: float = 0.0
    vmd_tol: float = 1e-7
    se_m: int = 2
    se_r: float = 0.2
    se_threshold: float = 0.4
    ewt_modes: int = 8
    decomposition_context: int = 0
    min_context: int = 64
    feature_selection: str = "paper"
    feature_bank: str = "full"
    feature_scaler: str = "standard"
    base_boost: str = "ridge"
    scaler_scope: str = "train"
    smoke: bool = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(path):
    raw = pd.read_csv(path)
    if raw["Stn Id"].nunique() != 1 or raw["Stn Name"].iloc[0] != "Brentwood":
        raise ValueError("Expected the single Brentwood station used in the paper.")
    hours = raw["Hour (PST)"].astype(int)
    timestamp = pd.to_datetime(raw["Date"], format="%m/%d/%Y") + pd.to_timedelta(hours // 100, unit="h")
    columns = ["ETo (in)", "Precip (in)", "Sol Rad (Ly/day)", "Vap Pres (mBars)", "Air Temp (F)",
               "Rel Hum (%)", "Dew Point (F)", "Wind Speed (mph)", "Wind Dir (0-360)", "Soil Temp (F)"]
    frame = pd.DataFrame({name: pd.to_numeric(raw[col], errors="coerce").to_numpy()
                          for name, col in zip(VARIABLES, columns)}, index=pd.DatetimeIndex(timestamp))
    frame.index.name = "timestamp"
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    if frame.index.has_duplicates:
        raise ValueError("Duplicate hourly timestamps: resolve explicitly before training.")
    frame.sort_index(inplace=True)
    for col in ["AT", "DPT", "ST"]:
        frame[col] = (frame[col] - 32) * 5 / 9
    frame["WS"] *= 0.44704
    return frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="h", name="timestamp"))


def clean_data(frame, train_end, causal):
    fit = frame.iloc[:train_end] if causal else frame
    mean, std = fit.mean(), fit.std(ddof=1)
    mask = (frame - mean).abs() > 3 * std
    result = frame.mask(mask)
    if causal:
        result = result.ffill().fillna(fit.mask((fit - mean).abs() > 3 * std).median())
    else:
        coordinates = np.arange(len(frame))
        for col in VARIABLES:
            valid = result[col].notna().to_numpy()
            if valid.sum() < 4:
                raise ValueError(f"Insufficient valid values for cubic interpolation: {col}")
            spline = CubicSpline(coordinates[valid], result.loc[valid, col].to_numpy(), bc_type="not-a-knot")
            result.loc[~valid, col] = spline(coordinates[~valid])
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Cleaning left nonfinite data.")
    return result, {"missing_before": frame.isna().sum().to_dict(), "outliers": mask.sum().to_dict(),
                    "fit_scope": "training only" if causal else "full season",
                    "imputation": "forward fill, training median fallback" if causal else "cubic spline, not-a-knot"}


def select_features(frame):
    rows, selected = [], ["WS"]
    for col in VARIABLES:
        if col == "WS":
            continue
        if frame[col].std() == 0 or frame["WS"].std() == 0:
            r, p = None, None
        else:
            r, p = (float(v) for v in pearsonr(frame[col], frame["WS"]))
        take = r is not None and abs(r) > 0.5 and p < 0.05
        rows.append({"feature": col, "pcc": r, "p_value": p, "selected": take})
        if take:
            selected.append(col)
    return selected, rows


def make_scaler(config):
    return StandardScaler() if config.feature_scaler == "standard" else MinMaxScaler()


def feature_matrix(frame, selected, components):
    meteo = [col for col in selected if col != "WS"]
    return np.column_stack([frame[meteo].to_numpy(), components, frame[["WS"]].to_numpy()])


def split_masks(origins, config, train_end, val_end):
    return [(origins + config.horizons <= train_end),
            (origins >= train_end) & (origins + config.horizons <= val_end), (origins >= val_end)]


def bank_features(frame, selected, k, config, fixed_high=None):
    parts, report, modes = decompose(frame["WS"].to_numpy(), k, config, fixed_high=fixed_high,
                                     return_modes=config.feature_bank == "full")
    if config.feature_bank == "full":
        low = [i for i in range(k) if i not in report["high_indices"]]
        ewt = parts[:, len(low):]
        ewt_labels = report["labels"][len(low):]
        meteo = [col for col in selected if col != "WS"]
        features = np.column_stack([frame[meteo].to_numpy(), modes.T, ewt, frame[["WS"]].to_numpy()])
        labels = meteo + [f"IMF{i + 1}" for i in range(k)] + ewt_labels + ["WS"]
    else:
        features, labels = feature_matrix(frame, selected, parts), [col for col in selected if col != "WS"] + report["labels"] + ["WS"]
    return features, labels, report


def full_season_windows(frame, selected, k, config, train_end, val_end):
    features, labels, report = bank_features(frame, selected, k, config)
    scaler = make_scaler(config).fit(features if config.scaler_scope == "full" else features[:train_end])
    scaled = scaler.transform(features)
    windows = np.lib.stride_tricks.sliding_window_view(scaled, config.window, axis=0).transpose(0, 2, 1)
    origins = np.arange(config.window, len(frame) - config.horizons + 1)
    xs = windows[origins - config.window].astype(np.float32)
    raw_targets = frame["WS"].to_numpy()[origins[:, None] + np.arange(config.horizons)]
    ys = scaled[origins] if config.architecture == "author" else raw_targets.astype(np.float32)
    masks = split_masks(origins, config, train_end, val_end)
    blocks = [(xs[m], ys[m], raw_targets[m], origins[m]) for m in masks]
    return blocks, scaler, report, labels


def causal_windows(frame, selected, k, config, train_end, val_end):
    _, train_report = decompose(frame["WS"].iloc[:train_end].to_numpy(), k, config)
    high = train_report["high_indices"]
    meteo = [col for col in selected if col != "WS"]
    if config.feature_bank == "full":
        low = [i for i in range(k) if i not in high]
        labels = meteo + [f"IMF{i + 1}" for i in range(k)] + train_report["labels"][len(low):] + ["WS"]
    else:
        labels = meteo + train_report["labels"] + ["WS"]
    origins = np.arange(max(config.window, config.min_context), len(frame) - config.horizons + 1)
    matrices = np.empty((len(origins), config.window, len(labels)), dtype=np.float32)
    raw_targets = frame["WS"].to_numpy()[origins[:, None] + np.arange(config.horizons)]
    feature_targets = np.zeros((len(origins), len(labels)), dtype=np.float32)
    last_report = None
    for i, origin in enumerate(origins):
        start = max(0, origin - config.decomposition_context) if config.decomposition_context else 0
        history = frame.iloc[start:origin]
        features, _, last_report = bank_features(history, selected, k, config, fixed_high=high)
        matrices[i] = features[-config.window:]
        if config.architecture == "author":
            target_history = frame.iloc[start:origin + 1]
            target_parts, _ = decompose(target_history["WS"].to_numpy(), k, config, fixed_high=high)
            feature_targets[i] = feature_matrix(target_history, selected, target_parts)[-1]
        if i == 0 or (i + 1) % 50 == 0 or i == len(origins) - 1:
            print(f"    causal decomposition {i + 1}/{len(origins)}, last observation={frame.index[origin - 1]}", flush=True)
    masks = split_masks(origins, config, train_end, val_end)
    scaler = make_scaler(config).fit(matrices[masks[0]].reshape(-1, len(labels)))
    xs = scaler.transform(matrices.reshape(-1, len(labels))).reshape(matrices.shape).astype(np.float32)
    ys = scaler.transform(feature_targets) if config.architecture == "author" else raw_targets.astype(np.float32)
    report = {"training_mode_selection": train_report, "last_origin": last_report, "causal": True,
              "decomposition_context": config.decomposition_context,
              "discarded_boundary_windows": int(sum(~(masks[0] | masks[1] | masks[2])))}
    return [(xs[m], ys[m].astype(np.float32), raw_targets[m], origins[m]) for m in masks], scaler, report, labels


def metrics(actual, predicted):
    diff = predicted - actual
    if np.any(actual == 0):
        raise ValueError("MAPE Eq. (28) undefined at zero wind speed; no silent epsilon substitution.")
    return {"rmse": float(np.sqrt(np.mean(diff ** 2))), "mae": float(np.mean(np.abs(diff))),
            "mape": float(100 * np.mean(np.abs(diff / actual)))}


def inverse_feature_target(prediction, scaler):
    if isinstance(scaler, StandardScaler):
        return prediction * scaler.scale_[-1] + scaler.mean_[-1]
    return (prediction - scaler.min_[-1]) / scaler.scale_[-1]


def select_vmd_parameters(frame, selected, config, train_end, val_end, window_function, out, season):
    """For each alpha minimize VMD RER over K, then minimize ridge validation RMSE."""
    if not 2 <= config.vmd_k_min <= config.vmd_k_max:
        raise ValueError("VMD K bounds must satisfy 2 <= vmd_k_min <= vmd_k_max.")
    ws = frame["WS"].to_numpy(dtype=float)
    if len(ws) < max(16, 2 * config.vmd_k_max) or not np.isfinite(ws).all():
        raise ValueError("K search needs a finite full-season signal with at least max(16, 2*Kmax) observations.")
    alphas = ALPHA_GRID if config.tune_alpha else (config.vmd_alpha,)
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Every VMD alpha candidate must be finite and positive.")
    k_candidates = list(range(config.vmd_k_min, config.vmd_k_max + 1))
    k_trials, alpha_trials = [], []
    # Persist partial diagnostics on failure, but never choose from an incomplete grid.
    try:
        for alpha in alphas:
            candidate = replace(config, vmd_alpha=float(alpha))
            current_trials = []
            for k in k_candidates:
                trial = {"alpha": float(alpha), "k": k, "rer_fraction": None,
                         "rer_percent": None, "best_k_for_alpha": False, "selected_pair": False,
                         "status": "running", "error": ""}
                k_trials.append(trial)
                try:
                    modes = vmd_modes(ws, k, candidate)
                    if modes.shape != (k, len(ws)):
                        raise ValueError(f"Unexpected VMD mode shape for alpha={alpha}, K={k}: {modes.shape}")
                    rer = relative_reconstruction_error(ws, modes)
                    trial.update(rer_fraction=rer, rer_percent=100 * rer, status="ok")
                except Exception as error:
                    trial.update(status="failed", error=str(error))
                    raise
                current_trials.append(trial)
                print(f"    reconstruction alpha={alpha:g}, K={k}: RER={100 * rer:.6f}%", flush=True)
            # Evaluate every K, even after RER drops below 3%. Exact ties prefer smaller K.
            best_k = min(current_trials, key=lambda row: (row["rer_fraction"], row["k"]))
            best_k["best_k_for_alpha"] = True
            k = best_k["k"]
            alpha_trial = {"alpha": float(alpha), "selected_k": k,
                           "rer_fraction": best_k["rer_fraction"], "rer_percent": best_k["rer_percent"],
                           "val_rmse": None, "selected_pair": False, "status": "running", "error": ""}
            alpha_trials.append(alpha_trial)
            try:
                blocks, _, _, _ = window_function(frame, selected, k, candidate, train_end, val_end)
                train_x, _, train_y, _ = blocks[0]
                val_x, _, val_y, _ = blocks[1]
                if not len(train_x) or not len(val_x):
                    raise ValueError("Empty training or validation split during VMD parameter selection.")
                # The third block entry contains raw WS targets for all three horizons.
                # No test targets are passed to the probe or used in its RMSE score.
                # Match the baseline paper-mode probe's float32 raw-WS target convention.
                train_y = np.asarray(train_y, dtype=np.float32)
                val_y = np.asarray(val_y, dtype=np.float32)
                probe = Ridge(alpha=0.1).fit(train_x.reshape(len(train_x), -1), train_y)
                prediction = probe.predict(val_x.reshape(len(val_x), -1))
                error = float(np.sqrt(np.mean((prediction - val_y) ** 2)))
                if not np.isfinite(error):
                    raise FloatingPointError("Nonfinite ridge validation RMSE.")
                alpha_trial.update(val_rmse=error, status="ok")
            except Exception as error:
                alpha_trial.update(status="failed", error=str(error))
                raise
            print(f"    alpha={alpha:g}, reconstruction-selected K={k}: validation RMSE={error:.6f}", flush=True)
        # This comparison is across one reconstruction-selected K per alpha.
        winner = min(alpha_trials, key=lambda row: (row["val_rmse"], row["alpha"]))
        winner["selected_pair"] = True
        for row in k_trials:
            row["selected_pair"] = row["alpha"] == winner["alpha"] and row["k"] == winner["selected_k"]
        report = {"search_complete": True,
                  "method": "minimum_VMD_RER_K_then_minimum_ridge_validation_RMSE_alpha",
                  "scope": "full cleaned available season; retrospective leakage protocol",
                  "k_candidates": k_candidates, "alpha_candidates": list(alphas),
                  "k_tie_break": "smaller K on exactly equal RER",
                  "alpha_tie_break": "smaller alpha on exactly equal validation RMSE",
                  "three_percent_used_for_selection": False, "ridge_penalty": 0.1,
                  "validation_target": "raw wind speed, all three horizons, RMSE in m/s",
                  "selected_k": winner["selected_k"], "selected_alpha": winner["alpha"],
                  "selected_rer_fraction": winner["rer_fraction"], "selected_rer_percent": winner["rer_percent"],
                  "selected_val_rmse": winner["val_rmse"],
                  "selected_k_at_search_boundary": winner["selected_k"] in (k_candidates[0], k_candidates[-1]),
                  "k_trials": k_trials, "alpha_trials": alpha_trials}
        return winner["selected_k"], winner["alpha"], report
    finally:
        pd.DataFrame(k_trials, columns=["alpha", "k", "rer_fraction", "rer_percent", "best_k_for_alpha",
                                       "selected_pair", "status", "error"]).to_csv(
            out / f"{season}_k_rer_search.csv", index=False)
        pd.DataFrame(alpha_trials, columns=["alpha", "selected_k", "rer_fraction", "rer_percent", "val_rmse",
                                           "selected_pair", "status", "error"]).to_csv(
            out / f"{season}_alpha_validation_search.csv", index=False)


def train(blocks, config, device, out, season, scaler):
    train_x, train_y, _, _ = blocks[0]
    val_x, val_y, _, _ = blocks[1]
    if not len(train_x) or not len(val_x) or not len(blocks[2][0]):
        raise ValueError("Empty split after constructing windows.")
    seed_all(config.seed)
    model = KANInformer(train_x.shape[-1], config, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr / 20)
    boost = config.base_boost == "ridge" and config.architecture == "paper"
    base, target_scaler = None, MinMaxScaler()
    train_targets, model_val_targets = train_y, val_y
    if config.architecture == "paper":
        if boost:
            base = Ridge(alpha=0.1).fit(train_x.reshape(len(train_x), -1), train_y)
            train_targets = train_y - base.predict(train_x.reshape(len(train_x), -1))
            model_val_targets = val_y - base.predict(val_x.reshape(len(val_x), -1))
        target_scaler.fit(train_targets)
        train_targets = target_scaler.transform(train_targets).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_targets)),
                        batch_size=config.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    validation = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(model_val_targets)),
                            batch_size=config.batch_size)
    val_x_tensor = torch.from_numpy(val_x)
    best, best_weights, stale, history = float("inf"), None, 0, []
    print(f"    model={config.architecture}, parameters={sum(p.numel() for p in model.parameters()):,}; "
          f"boost={'ridge' if boost else 'none'}; windows train/val/test={[len(b[0]) for b in blocks]}", flush=True)
    for epoch in range(config.epochs):
        model.train()
        loss_total = torch.zeros((), device=device)
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.detach() * len(x)
            if step == 0 or (step + 1) % 20 == 0:
                print(f"    epoch {epoch + 1}/{config.epochs}, batch {step + 1}/{len(loader)}, loss={loss.detach().item():.6f}", flush=True)
        scheduler.step()
        model.eval()
        if config.architecture == "paper":
            with torch.no_grad():
                prediction = target_scaler.inverse_transform(model.forecast(val_x_tensor).cpu().numpy())
            if boost:
                prediction = prediction + base.predict(val_x.reshape(len(val_x), -1))
            val_loss = float(np.mean((prediction - val_y) ** 2))
        else:
            val_total = torch.zeros((), device=device)
            with torch.no_grad():
                for x, y in validation:
                    x, y = x.to(device), y.to(device)
                    val_total += torch.nn.functional.mse_loss(model(x), y) * len(x)
            val_loss = val_total.item() / len(val_x)
        history.append({"epoch": epoch + 1, "train_mse": loss_total.item() / len(train_x), "val_mse": val_loss})
        print(f"    epoch {epoch + 1}: train={history[-1]['train_mse']:.6f}, val={val_loss:.6f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                print("    early stopping; restoring best validation checkpoint", flush=True)
                break
    model.load_state_dict(best_weights)
    checkpoint = {"model_state": best_weights, "config": asdict(config),
                  "feature_scaler": scaler.__class__.__name__, "feature_scale": scaler.scale_.tolist(),
                  "feature_offset": (scaler.mean_ if isinstance(scaler, StandardScaler) else scaler.min_).tolist()}
    if config.architecture == "paper":
        checkpoint["target_scale"] = target_scaler.scale_.tolist()
        checkpoint["target_offset"] = target_scaler.min_.tolist()
    if boost:
        checkpoint["base_coef"] = base.coef_.tolist()
        checkpoint["base_intercept"] = base.intercept_.tolist()
    torch.save(checkpoint, out / f"{season}_checkpoint.pt")
    pd.DataFrame(history).to_csv(out / f"{season}_training.csv", index=False)
    test_x, _, actual, origins = blocks[2]
    predicted = np.empty_like(actual)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_x), config.batch_size):
            x = torch.from_numpy(test_x[start:start + config.batch_size]).to(device)
            prediction = model.forecast(x).cpu().numpy()
            if config.architecture == "paper":
                prediction = target_scaler.inverse_transform(prediction)
                if boost:
                    prediction = prediction + base.predict(test_x[start:start + len(x)].reshape(len(x), -1))
            else:
                prediction = inverse_feature_target(prediction, scaler)
            predicted[start:start + len(x)] = prediction
    records = []
    for horizon in range(config.horizons):
        records.extend({"origin_index": int(origin), "horizon": horizon + 1, "actual": float(a), "prediction": float(p)}
                       for origin, a, p in zip(origins, actual[:, horizon], predicted[:, horizon]))
    pd.DataFrame(records).to_csv(out / f"{season}_predictions.csv", index=False)
    return [{"pipeline": PIPELINE_NAME, "season": season, "horizon": h + 1,
             **metrics(actual[:, h], predicted[:, h])} for h in range(config.horizons)]


def run(config, path, out, seasons, require_complete=False):
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pipeline: {PIPELINE_NAME}; CUDA available: {torch.cuda.is_available()}; device: {device}", flush=True)
    print("Execution of published methods; exact numerical reproduction is unverified. Read REPRODUCTION_AUDIT.md.", flush=True)
    raw = load_data(path)
    summaries, all_metrics = {}, []
    for season in seasons:
        start, end = PERIODS[season]
        frame = raw.loc[(raw.index >= start) & (raw.index < end)].copy()
        expected = len(pd.date_range(start, end, freq="h", inclusive="left"))
        complete = len(frame) == expected and not frame.isna().all(axis=1).any()
        print(f"\n{season}: supplied {len(frame)}/{expected} expected hourly rows; complete={complete}", flush=True)
        if require_complete and not complete:
            raise ValueError(f"{season}: supplied CSV does not cover the paper season. No synthetic tail will be created.")
        if config.smoke:
            frame = frame.iloc[:240]
        train_end, val_end = int(len(frame) * 0.8), int(len(frame) * 0.9)
        frame, cleaning = clean_data(frame, train_end, causal=not DECOMPOSE_BEFORE_SPLIT)
        pcc_frame = frame if DECOMPOSE_BEFORE_SPLIT else frame.iloc[:train_end]
        selected, pcc = select_features(pcc_frame)
        computed = list(selected)
        if config.feature_selection == "paper":
            selected = PAPER_INPUTS[season]
        print(f"    computed PCC inputs={computed}; used={selected}; reported paper inputs={PAPER_INPUTS[season]}", flush=True)
        begin = time.perf_counter()
        function = full_season_windows if DECOMPOSE_BEFORE_SPLIT else causal_windows
        k, selected_alpha, selection = select_vmd_parameters(
            frame, selected, config, train_end, val_end, function, out, season)
        season_config = replace(config, vmd_alpha=selected_alpha)
        (out / f"{season}_vmd_selection.json").write_text(
            json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8")
        print(f"    selected alpha={selected_alpha:g}, K={k}, RER={selection['selected_rer_percent']:.6f}%", flush=True)
        blocks, scaler, decomposition, labels = function(frame, selected, k, season_config, train_end, val_end)
        if not np.isclose(decomposition["vmd_relative_absolute_residual"], selection["selected_rer_fraction"],
                          rtol=1e-9, atol=1e-12):
            raise RuntimeError("Final VMD reconstruction differs from the selected candidate; training will not start.")
        print(f"    decomposition complete in {time.perf_counter() - begin:.1f}s; features={labels}", flush=True)
        rows = train(blocks, season_config, device, out, season, scaler)
        all_metrics.extend(rows)
        pd.DataFrame(pcc).to_csv(out / f"{season}_pcc.csv", index=False)
        summaries[season] = {"complete_paper_season": complete, "expected_hours": expected, "rows_used": len(frame),
                             "cleaning": cleaning, "selected_inputs": selected, "computed_inputs": computed,
                             "paper_inputs_match": set(computed) == set(PAPER_INPUTS[season]),
                             "reference_paper_k": PAPER_K[season], "vmd_k_selected": k,
                             "vmd_alpha_selected": selected_alpha, "vmd_alpha_trials": selection["alpha_trials"],
                             "vmd_selection": selection,
                             "decomposition": decomposition, "feature_columns": labels}
        for row in rows:
            print(f"    h{row['horizon']}: RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f}, MAPE={row['mape']:.3f}%", flush=True)
        pd.DataFrame(all_metrics).to_csv(out / "metrics.csv", index=False)
    payload = {"pipeline": PIPELINE_NAME, "config": asdict(config), "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "exact_replication": False, "decompose_before_split": DECOMPOSE_BEFORE_SPLIT,
               "reports": summaries, "metrics": all_metrics, "torch_version": torch.__version__, "device": str(device)}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["| Season | Horizon | Paper RMSE | Run RMSE | Paper MAE | Run MAE | Paper MAPE | Run MAPE |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in all_metrics:
        p = PAPER_METRICS[row["season"]][row["horizon"] - 1]
        lines.append(f"| {row['season']} | h{row['horizon']} | {p[0]:.3f} | {row['rmse']:.6f} | {p[1]:.3f} | {row['mae']:.6f} | {p[2]:.1f}% | {row['mape']:.3f}% |")
    (out / "paper_format_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Full-season KANInformer: reconstruction selects K, ridge validation selects alpha.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/hourly.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--seasons", nargs="+", choices=SEASONS, default=SEASONS)
    parser.add_argument("--require-paper-data", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    defaults = Config()
    for name, value in asdict(defaults).items():
        if name in ("smoke", "tune_alpha"):
            continue
        choices = {"architecture": ["paper", "author"], "feature_selection": ["pcc", "paper"],
                   "feature_bank": ["caewt", "full"], "feature_scaler": ["minmax", "standard"],
                   "base_boost": ["none", "ridge"], "scaler_scope": ["train", "full"]}.get(name)
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value, choices=choices)
    parser.add_argument("--tune-alpha", action=argparse.BooleanOptionalAction, default=defaults.tune_alpha,
                        help="compare alpha candidates by ridge validation RMSE after minimizing RER over K for each alpha")
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in asdict(defaults) if name not in ("smoke", "tune_alpha")},
                    tune_alpha=args.tune_alpha)
    if config.d_model % config.n_heads or config.d_model % 2:
        parser.error("d-model must be even and divisible by n-heads")
    if not 1 <= config.label_len <= config.window:
        parser.error("label-len must be within the input window")
    if config.epochs < 1 or config.batch_size < 1 or config.min_context < 16:
        parser.error("epochs/batch-size must be positive and min-context must be at least 16")
    if config.decomposition_context and config.decomposition_context < config.min_context:
        parser.error("decomposition-context must be zero or at least min-context")
    if not DECOMPOSE_BEFORE_SPLIT and config.scaler_scope != "train":
        parser.error("no_dataleak requires training-only scaler fitting")
    if not config.tune_alpha and config.vmd_alpha <= 0:
        parser.error("vmd-alpha must be positive when --no-tune-alpha is set")
    if not np.isfinite(config.vmd_alpha) or config.vmd_alpha <= 0:
        parser.error("vmd-alpha must be finite and positive")
    if not 2 <= config.vmd_k_min <= config.vmd_k_max:
        parser.error("VMD K bounds must satisfy 2 <= vmd-k-min <= vmd-k-max")
    if config.architecture == "author" and config.base_boost != "none":
        parser.error("the author architecture trains next-feature targets and does not support --base-boost")
    if config.architecture == "author" and config.feature_bank != "caewt":
        parser.error("the author architecture requires --feature-bank caewt so target widths match")
    if not args.skip_smoke or args.smoke_only:
        from checks import MethodChecks, SelectionChecks
        threads = torch.get_num_threads()
        try:
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                                       for case in (MethodChecks, SelectionChecks))
            verified = unittest.TextTestRunner(verbosity=1).run(suite)
        finally:
            torch.set_num_threads(threads)
        if not verified.wasSuccessful():
            raise RuntimeError("Method checks failed; full training will not start.")
        smoke = replace(config, smoke=True, epochs=1, d_model=8, n_heads=2, kan_hidden=4, ewt_modes=3,
                        min_context=32, decomposition_context=64, vmd_tol=1e-4)
        smoke_dir = args.output / "_smoke"
        if smoke_dir.exists():
            parser.error(f"Existing smoke outputs at {smoke_dir}; choose another --output")
        run(smoke, args.data, smoke_dir, [args.seasons[0]])
        resolved = smoke_dir.resolve()
        if resolved.parent != args.output.resolve() or resolved.name != "_smoke":
            raise RuntimeError("Unsafe smoke cleanup path")
        shutil.rmtree(resolved)
        print("Smoke passed; smoke artifacts removed.", flush=True)
        if args.smoke_only:
            return
    run(config, args.data, args.output, args.seasons, args.require_paper_data)


if __name__ == "__main__":
    main()
