"""Compare only matched Brentwood forecasts from L, L.1 and spatial_L.1."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
EXPERIMENTS=["L","L.1","spatial_L.1"]


def summarize(run_id,root=ROOT):
    folders={e:root/e/"results"/run_id for e in EXPERIMENTS}
    protocols={e:json.loads((folder/"protocol.json").read_text()) for e,folder in folders.items()}
    reference=protocols["L.1"]
    for experiment,p in protocols.items():
        for field in ["year","seasons","data_hash"]:
            if p[field]!=reference[field]: raise ValueError(f"Cannot compare mismatched {field}: {experiment}")
        for key in set(p["config"]) & set(reference["config"]):
            if p["config"][key]!=reference["config"][key]:
                raise ValueError(f"Cannot compare different training settings: {experiment}, {key}")
    comparison=[]
    for season in reference["seasons"]:
        predictions={e:pd.read_csv(f/f"{season}_predictions.csv").sort_values(["origin_index","horizon"]).reset_index(drop=True)
                     for e,f in folders.items()}
        ref=predictions["L.1"]
        for e,p in predictions.items():
            for c in ["origin_index","horizon","origin_time","target_time","station"]:
                if not p[c].equals(ref[c]): raise ValueError(f"Different scored timestamps/stations: {season}, {e}, {c}")
            np.testing.assert_allclose(p.actual,ref.actual,rtol=0,atol=1e-12,
                                       err_msg=f"Different ground truth in {season}, {e}")
        # Scores are loaded only after matching predictions/ground truth.
        for e,f in folders.items():
            metrics=pd.read_csv(f/"metrics.csv")
            comparison.extend(metrics[metrics.season==season].to_dict("records"))
    table=pd.DataFrame(comparison)
    out=root/"results"/run_id
    out.mkdir(parents=True,exist_ok=False)
    table.to_csv(out/"brentwood_metrics_long.csv",index=False)
    parameters=[]
    for experiment,folder in folders.items():
        summary=json.loads((folder/"summary.json").read_text())
        for season,report in summary["reports"].items():
            selection=report["Brentwood"]["selection"]
            parameters.append({"experiment":experiment,"year":reference["year"],"season":season,
                "k":selection["selected_k"],"alpha":selection["selected_alpha"]})
    pd.DataFrame(parameters).to_csv(out/"selected_parameters.csv",index=False)
    wide=table.pivot(index=["year","season","horizon"],columns="experiment",values=["rmse","mae","mape","mape_nonzero"])
    wide.to_csv(out/"brentwood_comparison.csv")
    lines=["# Brentwood-only matched comparison","",
        "L: prediction-selected alpha at fixed seasonal K. L.1: reconstruction-guided K and prediction-selected alpha.",
        "Spatial L.1: L.1 plus neighboring-station graph context, with Brentwood-only Ridge and supervision.","",
        "All scores use the same cleaned Brentwood targets and timestamps. Full-season leakage is retained.",
        "Strict MAPE is NA if any target equals zero. MAPE_nonzero is a separately labelled diagnostic excluding exact zero targets.",
        "Positive improvement means spatial L.1 has a lower error than the named baseline. Bold retains exact-numerical ties.","",
        "|Season|Horizon|Metric|L|L.1|Spatial L.1|Improvement vs L (%)|Improvement vs L.1 (%)|",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    winners=[];improvements=[]
    def fmt(v): return "NA" if pd.isna(v) else f"{v:.6f}"
    for season in reference["seasons"]:
        group=table[table.season==season]
        for h in sorted(group.horizon.unique()):
            row=group[group.horizon==h].set_index("experiment")
            for metric in ["rmse","mae","mape","mape_nonzero"]:
                scores=row.loc[EXPERIMENTS,metric]
                best=scores.min() if scores.notna().any() else np.nan
                values=[]
                for e in EXPERIMENTS:
                    value=scores[e];s=fmt(value)
                    if pd.notna(value) and np.isclose(value,best,rtol=1e-10,atol=1e-12):
                        s=f"**{s}**";winners.append({"season":season,"horizon":h,"metric":metric,"winner":e,"value":value})
                    values.append(s)
                changes=[]
                for e in ["L","L.1"]:
                    baseline,spatial=scores[e],scores["spatial_L.1"]
                    pct=100*(baseline-spatial)/baseline if pd.notna(baseline) and pd.notna(spatial) and baseline!=0 else np.nan
                    changes.append(fmt(pct))
                    improvements.append({"season":season,"horizon":h,"metric":metric,"baseline":e,
                        "spatial_improvement_pct":pct})
                lines.append(f"|{season}|{h}|{metric}|"+"|".join(values+changes)+"|")
    (out/"comparison.md").write_text("\n".join(lines),encoding="utf-8")
    pd.DataFrame(winners).to_csv(out/"best_results.csv",index=False)
    pd.DataFrame(improvements).to_csv(out/"spatial_improvements.csv",index=False)
    pd.DataFrame(winners).groupby(["metric","winner"]).size().rename("winning_rows").to_csv(out/"win_counts.csv")
    note={"target_station":47,"matched_timestamps_and_targets":True,"year":reference["year"],
        "experiments":EXPERIMENTS,"source_folders":{e:str(f) for e,f in folders.items()},
        "devices":{e:p["device"] for e,p in protocols.items()},
        "warning":"Single recorded seed; leakage-based seasonal experiment. No conclusion of operational skill or statistical significance."}
    (out/"comparison_audit.json").write_text(json.dumps(note,indent=2),encoding="utf-8")
    print(f"Final comparison: {out/'comparison.md'}")
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id",required=True)
    a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in (".","..") or any(c in a.run_id for c in ':\\/'):
        p.error("run-id must be a plain folder name")
    summarize(a.run_id)
