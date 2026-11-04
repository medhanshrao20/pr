"""Separate adapters. Protected L/L.1 source files and baseline functions are never edited."""
import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prepare_data import prepare, LEGACY_COLUMNS
from .graph_model import TargetGraphKAN, target_ridge_class
from .spatial_graph_utilities import build_graph

ROOT=Path(__file__).resolve().parents[1]
VARIABLES=["ET","PCP","SR","VP","AT","RH","DPT","WS","WD","ST"]
ZERO_MAPE_ERROR="MAPE Eq. (28) undefined at zero wind speed; no silent epsilon substitution."


def write_json(path,data):
    def convert(x):
        if isinstance(x,Path): return str(x)
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(data,indent=2,default=convert,allow_nan=False),encoding="utf-8")


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_engine(kind):
    source=ROOT/("L" if kind=="L" else "L.1")/"source"
    # Original absolute imports are resolved to local copied files, then restored.
    previous={k:sys.modules.get(k) for k in ("model","decomposition")}
    name="comparison_L" if kind=="L" else "comparison_L1"
    try:
        sys.modules["model"]=load_module(name+"_model",source/"model.py")
        sys.modules["decomposition"]=load_module(name+"_decomposition",source/"decomposition.py")
        return load_module(name+"_run",source/"run.py")
    finally:
        for k,v in previous.items():
            if v is None: sys.modules.pop(k,None)
            else: sys.modules[k]=v


def seasonal_periods(year):
    return {"spring":(f"{year}-03-01",f"{year}-06-01"),
        "summer":(f"{year}-06-01",f"{year}-09-01"),
        "autumn":(f"{year}-09-01",f"{year}-12-01"),
        "winter":(f"{year-1}-12-01",f"{year}-03-01")}


def load_other_station(path):
    # Same numeric mapping and conversions as the copied Brentwood loader, without its identity restriction.
    raw=pd.read_csv(path)
    if raw["Stn Id"].nunique()!=1: raise ValueError("Expected one station in a derived CSV.")
    timestamp=pd.to_datetime(raw.Date,format="%m/%d/%Y")+pd.to_timedelta(raw["Hour (PST)"].astype(int)//100,unit="h")
    frame=pd.DataFrame({v:pd.to_numeric(raw[c],errors="coerce").to_numpy() for v,c in zip(VARIABLES,LEGACY_COLUMNS)},index=pd.DatetimeIndex(timestamp))
    frame.index.name="timestamp"
    if frame.index.has_duplicates: raise ValueError("Duplicate neighbor timestamps.")
    frame=frame.sort_index().replace([np.inf,-np.inf],np.nan)
    for c in ("AT","DPT","ST"): frame[c]=(frame[c]-32)*5/9
    frame.WS*=.44704
    return frame.reindex(pd.date_range(frame.index.min(),frame.index.max(),freq="h",name="timestamp"))


def choose_year(frames,seasons,requested=None):
    def covers(y):
        return all(f.index.min()<=pd.Timestamp(seasonal_periods(y)[s][0]) and
                   f.index.max()>=pd.Timestamp(seasonal_periods(y)[s][1])-pd.Timedelta(hours=1)
                   for f in frames.values() for s in seasons)
    if requested is not None:
        if not covers(requested): raise ValueError("Requested year/seasons are not fully covered by every selected station.")
        return requested
    latest=min(f.index.max().year for f in frames.values())
    earliest=max(f.index.min().year for f in frames.values())
    for year in range(latest,earliest-1,-1):
        if covers(year): return year
    raise ValueError("No complete requested seasonal cycle shared by the selected stations.")


def scaler_record(scaler):
    return {"class":type(scaler).__name__,"scale":scaler.scale_.tolist(),
        "offset":(scaler.mean_ if hasattr(scaler,"mean_") else scaler.min_).tolist()}


def select_parameters(engine,kind,frame,selected,config,tr,va,out,label):
    if kind=="L":
        season=label.split("_station_")[0]
        k=engine.PAPER_K[season]
        alpha,trials=engine.select_alpha(frame,selected,k,config,tr,va,engine.full_season_windows)
        return k,alpha,{"method":"fixed_published_K_then_validation_RMSE_alpha","selected_k":k,
            "selected_alpha":alpha,"alpha_trials":trials}
    return engine.select_vmd_parameters(frame,selected,config,tr,va,engine.full_season_windows,out,label)


def prepare_station(engine,kind,raw,season,year,config,out,label,selection_override=None):
    start,end=seasonal_periods(year)[season]
    times=pd.date_range(start,end,freq="h",inclusive="left",name="timestamp")
    frame=raw.reindex(times)
    tr,va=int(len(frame)*.8),int(len(frame)*.9)
    cleaned,cleaning=engine.clean_data(frame,tr,causal=False)
    computed,pcc=engine.select_features(cleaned)
    selected=engine.PAPER_INPUTS[season] if config.feature_selection=="paper" else computed
    if selection_override is None:
        k,alpha,selection=select_parameters(engine,kind,cleaned,selected,config,tr,va,out,label)
    else:
        k,alpha=selection_override["selected_k"],selection_override["selected_alpha"]
        selection=selection_override
    resolved=replace(config,vmd_alpha=alpha)
    blocks,scaler,decomposition,labels=engine.full_season_windows(cleaned,selected,k,resolved,tr,va)
    if "selected_rer_fraction" in selection and not np.isclose(decomposition["vmd_relative_absolute_residual"],
            selection["selected_rer_fraction"],rtol=1e-9,atol=1e-12):
        raise ValueError("Rebuilt decomposition differs from selected candidate.")
    pd.DataFrame(pcc).to_csv(out/f"{label}_pcc.csv",index=False)
    cleaned.to_csv(out/f"{label}_cleaned.csv")
    write_json(out/f"{label}_selection.json",selection)
    report={"selection":selection,"cleaning":cleaning,"features":labels,"selected_inputs":selected,
        "computed_inputs":computed,"decomposition":decomposition,"feature_scaler":scaler_record(scaler),
        "period_start":start,"period_end_exclusive":end,"rows":len(frame),"train_end":tr,"validation_end":va,
        "windows":[len(b[0]) for b in blocks]}
    return blocks,scaler,resolved,cleaned,report


def score_prediction_file(path):
    predictions=pd.read_csv(path)
    rows=[]
    for h,df in predictions.groupby("horizon"):
        y,p=df.actual.to_numpy(),df.prediction.to_numpy()
        if not np.isfinite(y).all() or not np.isfinite(p).all(): raise ValueError("Nonfinite saved forecasts/targets.")
        e=p-y; nonzero=y!=0
        strict_mape=float(100*np.mean(np.abs(e/y))) if nonzero.all() else None
        rows.append({"horizon":int(h),"rmse":float(np.sqrt(np.mean(e**2))),"mae":float(np.mean(np.abs(e))),
            "mape":strict_mape,"mape_nonzero":float(100*np.mean(np.abs(e[nonzero]/y[nonzero]))) if nonzero.any() else None,
            "n":len(y),"zero_targets":int((~nonzero).sum()),"mape_nonzero_n":int(nonzero.sum())})
    return rows


def call_protected_train(engine,blocks,config,device,out,season,scaler):
    try:
        engine.train(blocks,config,device,out,season,scaler)
        return None
    except ValueError as error:
        # The original train function saves checkpoint/history/predictions BEFORE its strict MAPE evaluation.
        # Preserve that behavior and report undefined strict MAPE externally; never modify engine.metrics.
        if str(error)!=ZERO_MAPE_ERROR or not (out/f"{season}_predictions.csv").is_file(): raise
        return str(error)


def argument_parser(experiment):
    p=argparse.ArgumentParser(description=f"{experiment}: protected copied code, new data, Brentwood-only evaluation.")
    p.add_argument("--data",type=Path,default=ROOT/"data")
    p.add_argument("--metadata",type=Path,default=ROOT/"data/station_coordinates.json")
    p.add_argument("--prepared-root",type=Path,default=ROOT/"prepared")
    p.add_argument("--year",type=int)
    p.add_argument("--seasons",nargs="+",choices=["spring","summer","autumn","winter"],default=["spring","summer","autumn","winter"])
    p.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    p.add_argument("--output",type=Path)
    for name in ["epochs","patience","batch-size","seed","d-model","n-heads","kan-hidden","kan-grid","ewt-modes"]:
        p.add_argument("--"+name,type=int,help="Optional override; omitted values retain copied Config defaults.")
    p.add_argument("--device",choices=["cpu","cuda","auto"],default="cpu")
    p.add_argument("--neighbor-stations",nargs="+",type=int,help="Spatial run: default all other 13 stations.")
    p.add_argument("--neighbors",type=int,default=3)
    p.add_argument("--graph",choices=["geographic","statistical","combined"],default="geographic")
    return p


def execute(experiment,args):
    if experiment not in ("L","L.1","spatial_L.1"): raise ValueError("Unknown comparison experiment.")
    if Path(args.run_id).name!=args.run_id or args.run_id in (".","..") or any(c in args.run_id for c in ':\\/'):
        raise ValueError("run-id must be a plain directory name.")
    out=args.output or ROOT/experiment/"results"/args.run_id
    if out.exists(): raise FileExistsError(f"Output already exists: {out}")
    kind="L" if experiment=="L" else "L.1"
    engine=load_engine(kind)
    overrides={name:getattr(args,name) for name in ("epochs","patience","batch_size","seed","d_model","n_heads","kan_hidden","kan_grid","ewt_modes") if getattr(args,name) is not None}
    config=replace(engine.Config(),**overrides)
    if min(config.epochs,config.patience,config.batch_size,config.kan_hidden,config.kan_grid,config.ewt_modes)<1:
        raise ValueError("Training dimensions/counts must be positive.")
    if config.n_heads<1 or config.d_model<2 or config.d_model%2 or config.d_model%config.n_heads:
        raise ValueError("d-model must be even and divisible by positive n-heads.")
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable.")
    if experiment!="spatial_L.1" and device.type=="cuda":
        raise ValueError("Protected baseline train() validates a CPU tensor without moving it to CUDA. Use --device cpu; no L/L.1 code is changed to fix this inherited issue.")
    files=sorted(args.data.glob("station_*.csv"))
    if not files: raise ValueError("No station files.")
    data_hash=hashlib.sha256(b"".join(p.name.encode()+hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest()
    conversion=prepare(args.data,args.prepared_root/data_hash[:16]/"legacy")
    if conversion.station.duplicated().any() or 47 not in conversion.station.to_list(): raise ValueError("Require unique station files including Brentwood 47.")
    legacy=args.prepared_root/data_hash[:16]/"legacy"
    raw={47:engine.load_data(legacy/"station_047.csv")}
    if experiment=="spatial_L.1":
        ids=args.neighbor_stations or sorted(set(conversion.station)-{47})
        if 47 in ids or len(set(ids))!=len(ids) or not set(ids).issubset(set(conversion.station)):
            raise ValueError("Neighbors must be distinct available IDs other than 47.")
        if not ids: raise ValueError("At least one other station is required.")
        for sid in sorted(ids): raw[sid]=load_other_station(legacy/f"station_{sid:03d}.csv")
    year=choose_year(raw,args.seasons,args.year)
    out.mkdir(parents=True)
    write_json(out/"arguments.json",vars(args))
    source=ROOT/("L" if kind=="L" else "L.1")/"source"
    provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    write_json(out/"protocol.json",{"experiment":experiment,"target_station":47,"source_kind":kind,
        "year":year,"seasons":args.seasons,"config":asdict(config),"data_hash":data_hash,
        "input_hashes":conversion.to_dict("records"),"copied_code_hashes":provenance,"device":str(device),
        "split":"80/10/10 separately within each complete calendar season",
        "leakage":"Original full-season 3-sigma cleaning, spline imputation, VMD/EWT before split retained.",
        "targets":"Cleaned Brentwood wind speed, as in the unchanged source; raw/imputed flags saved.",
        "baseline_source_modifications":False,"baseline_function_overrides":False,
        "strict_mape":"Original definition; undefined when any target equals zero.",
        "additional_mape_nonzero":"Separate external diagnostic, excludes exact zeros only; eligible counts reported."})
    station_ids=list(raw)
    meta=None
    if experiment=="spatial_L.1":
        meta=pd.DataFrame(json.loads(args.metadata.read_text(encoding="utf-8-sig"))).set_index("id").loc[station_ids]
        if not np.isfinite(meta[["lat","lon"]].to_numpy()).all(): raise ValueError("Invalid coordinate metadata.")
        meta.to_csv(out/"station_order_and_coordinates.csv",index_label="station")
    all_metrics=[]; reports={}
    print(f"{experiment}; target=Brentwood only; year={year}; device={device}; seed={config.seed}",flush=True)
    for season in args.seasons:
        override=None
        baseline_protocol=ROOT/"L.1/results"/args.run_id/"protocol.json"
        baseline_selection=ROOT/"L.1/results"/args.run_id/f"{season}_selection.json"
        if experiment=="spatial_L.1" and baseline_protocol.exists() and baseline_selection.exists():
            previous=json.loads(baseline_protocol.read_text())
            if previous["data_hash"]!=data_hash or previous["config"]!=asdict(config) or previous["year"]!=year:
                raise ValueError("Same-run L.1 configuration/data do not match the spatial experiment.")
            override=json.loads(baseline_selection.read_text())
        blocks,scaler,resolved,cleaned,own_report=prepare_station(engine,kind,raw[47],season,year,config,out,season,override)
        reports[season]={"Brentwood":own_report}
        train_blocks=blocks
        if experiment=="spatial_L.1":
            node_blocks=[blocks]; widths=[blocks[0][0].shape[-1]]; node_info={47:own_report}
            cleaned_nodes={47:cleaned}
            for sid in station_ids[1:]:
                other,other_scaler,other_config,other_cleaned,other_report=prepare_station(
                    engine,"L.1",raw[sid],season,year,config,out,f"{season}_station_{sid:03d}")
                for target_block,neighbor_block in zip(blocks,other):
                    np.testing.assert_array_equal(target_block[3],neighbor_block[3])
                node_blocks.append(other);widths.append(other[0][0].shape[-1])
                node_info[sid]=other_report;cleaned_nodes[sid]=other_cleaned
            training_wind=pd.DataFrame({sid:f.WS.iloc[:own_report["train_end"]] for sid,f in cleaned_nodes.items()})
            adjacency=build_graph(args.graph,meta,training_wind,args.neighbors)
            pd.DataFrame(adjacency,index=station_ids,columns=station_ids).to_csv(out/f"{season}_adjacency.csv")
            train_blocks=[(np.concatenate([node[j][0] for node in node_blocks],axis=-1),b[1],b[2],b[3]) for j,b in enumerate(blocks)]
            write_json(out/f"{season}_spatial_schema.json",{"station_order":station_ids,"node_feature_widths":widths,
                "node_reports":node_info,"graph":args.graph,"neighbors":args.neighbors,
                "ridge":"Brentwood own features only; unchanged local Ridge objective.",
                "supervision":"Brentwood targets only; no other station neural output/loss.",
                "brentwood_selection_reused":override is not None})
            reports[season]["nodes"]=node_info
            original_model,original_ridge=engine.KANInformer,engine.Ridge
            engine.KANInformer=lambda features,cfg,dev:TargetGraphKAN(original_model,widths,adjacency,cfg,dev)
            engine.Ridge=target_ridge_class(sum(widths),widths[0],config.window)
            try:
                notice=call_protected_train(engine,train_blocks,resolved,device,out,season,scaler)
            finally:
                engine.KANInformer,engine.Ridge=original_model,original_ridge
        else:
            # No source function/class replacements in either protected baseline.
            notice=call_protected_train(engine,blocks,resolved,device,out,season,scaler)
        predictions=pd.read_csv(out/f"{season}_predictions.csv")
        origins=predictions.origin_index.to_numpy(int);h=predictions.horizon.to_numpy(int)
        predictions["station"]=47
        predictions["origin_time"]=cleaned.index[origins-1].astype(str)
        predictions["target_time"]=cleaned.index[origins+h-1].astype(str)
        raw_actual=raw[47].reindex(cleaned.index).WS.to_numpy()[origins+h-1]
        predictions["raw_actual_before_cleaning"]=raw_actual
        predictions["target_changed_by_cleaning"]=(~np.isfinite(raw_actual))|(~np.isclose(raw_actual,predictions.actual))
        predictions.to_csv(out/f"{season}_predictions.csv",index=False)
        rows=score_prediction_file(out/f"{season}_predictions.csv")
        all_metrics.extend({"experiment":experiment,"station":47,"year":year,"season":season,**r} for r in rows)
        reports[season]["original_mape_notice"]=notice
        pd.DataFrame(all_metrics).to_csv(out/"metrics.csv",index=False)
        write_json(out/"summary.json",{"experiment":experiment,"year":year,"config":asdict(config),
            "data_hash":data_hash,"reports":reports,"metrics":all_metrics,"device":str(device)})
    print(f"Completed {experiment}: {out}",flush=True)
    return out


def main(experiment):
    args=argument_parser(experiment).parse_args()
    execute(experiment,args)
