"""Create legacy-schema CSV views of the copied new station data; no imputation/training."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE_COLUMNS=["eto_in","precip_in","sol_rad_ly_per_day","vapor_pressure_mbar","air_temp_f",
                "relative_humidity_pct","dew_point_f","wind_speed_mph","wind_direction_deg","soil_temp_f"]
LEGACY_COLUMNS=["ETo (in)","Precip (in)","Sol Rad (Ly/day)","Vap Pres (mBars)","Air Temp (F)",
                "Rel Hum (%)","Dew Point (F)","Wind Speed (mph)","Wind Dir (0-360)","Soil Temp (F)"]


def canonical_view(source):
    raw=pd.read_csv(source)
    required={"timestamp_pst","stn_id","stn_name",*SOURCE_COLUMNS}
    if not required.issubset(raw): raise ValueError(f"Missing input columns in {source}")
    if raw.stn_id.nunique()!=1: raise ValueError(f"Multiple station IDs: {source}")
    t=pd.to_datetime(raw.timestamp_pst,utc=True).dt.tz_convert("Etc/GMT+8").dt.tz_localize(None)
    if t.duplicated().any() or (t.dt.minute.ne(0)|t.dt.second.ne(0)).any():
        raise ValueError("Require unique whole-hour PST timestamps.")
    midnight=t.dt.hour.eq(0)
    date=t.dt.normalize()-pd.to_timedelta(midnight.astype(int),unit="D")
    view=pd.DataFrame({"Stn Id":raw.stn_id,"Stn Name":raw.stn_name,
        "CIMIS Region":raw.cimis_region if "cimis_region" in raw else "",
        "Date":date.dt.strftime("%m/%d/%Y"),"Hour (PST)":np.where(midnight,2400,t.dt.hour*100),
        "Jul":date.dt.dayofyear})
    for old,new in zip(SOURCE_COLUMNS,LEGACY_COLUMNS): view[new]=pd.to_numeric(raw[old],errors="coerce")
    return view


def prepare(data=ROOT/"data",destination=ROOT/"data/legacy"):
    destination.mkdir(parents=True,exist_ok=True)
    sources=sorted(data.glob("station_*.csv"))
    if not sources: raise ValueError(f"No station CSVs in {data}")
    rows=[]
    for source in sources:
        view=canonical_view(source)
        sid=int(view["Stn Id"].iloc[0])
        dest=destination/f"station_{sid:03d}.csv"
        payload=view.to_csv(index=False).encode("utf-8")
        # Refuse to silently reuse a stale/tampered derived view.
        if dest.exists() and dest.read_bytes()!=payload:
            raise ValueError(f"Derived CSV differs: {dest}. Use a fresh comparison copy for changed data.")
        if not dest.exists(): dest.write_bytes(payload)
        rows.append({"station":sid,"source":source.name,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "legacy_file":dest.name,"legacy_sha256":hashlib.sha256(payload).hexdigest(),"rows":len(view)})
    audit=pd.DataFrame(rows)
    audit.to_csv(destination/"conversion_audit.csv",index=False)
    return audit


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data")
    p.add_argument("--destination",type=Path,default=ROOT/"data/legacy")
    a=p.parse_args()
    print(prepare(a.data,a.destination).to_string(index=False))
