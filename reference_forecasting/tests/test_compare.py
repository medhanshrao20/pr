"""Synthetic tests only. Protected source files are not edited."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prepare_data import canonical_view,prepare,SOURCE_COLUMNS
from common.experiment import (load_engine,load_other_station,choose_year,argument_parser,execute,
                               call_protected_train,score_prediction_file,scaler_record)
from common.graph_model import TargetGraphKAN,target_ridge_class
from summarize import summarize


class CompareChecks(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    @staticmethod
    def make_data(root):
        data=Path(root)/"data";data.mkdir()
        times=pd.date_range("2025-03-01", "2025-06-01",freq="h",inclusive="left",tz="Etc/GMT+8")
        metadata=[]
        for sid,n in [(47,0),(2,1),(5,2)]:
            df=pd.DataFrame({"timestamp_pst":times.astype(str),"stn_id":sid,
                "stn_name":"Brentwood" if sid==47 else f"station_{sid}","cimis_region":"synthetic"})
            for j,c in enumerate(SOURCE_COLUMNS): df[c]=5+n+j+np.sin(np.arange(len(times))/(10+j+n))
            df.air_temp_f+=60;df.dew_point_f+=50;df.soil_temp_f+=50
            df.to_csv(data/f"station_{sid:03d}_test.csv",index=False)
            metadata.append({"id":sid,"name":df.stn_name.iloc[0],"lat":37+n,"lon":-121+n*.5})
        meta=data/"station_coordinates.json";meta.write_text(json.dumps(metadata))
        return data,meta

    def test_exact_source_hashes_and_function_locations(self):
        manifest=json.loads((ROOT/"source_manifest.json").read_text(encoding="utf-8-sig"))
        for path in (ROOT/"L/source").glob("*.py"):
            original=next(r for r in manifest if r["path"].endswith("\\original_dataleak\\"+path.name))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),original["sha256"])
        for path in (ROOT/"L.1/source").glob("*.py"):
            original=next(r for r in manifest if r["path"].endswith("\\original_dataleak.v1\\"+path.name))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),original["sha256"])
        for kind in ("L","L.1"):
            engine=load_engine(kind)
            self.assertEqual(engine.Ridge,Ridge)
            self.assertEqual(Path(engine.train.__code__.co_filename),ROOT/kind/"source/run.py")
            self.assertEqual(Path(engine.metrics.__code__.co_filename),ROOT/kind/"source/run.py")
            self.assertEqual(engine.Config().patience,30)
            self.assertEqual(engine.Config().epochs,300)

    def test_data_conversion_preserves_timestamps_and_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            data,meta=self.make_data(temp)
            audit=prepare(data,Path(temp)/"derived")
            self.assertEqual(len(audit),3)
            engine=load_engine("L.1")
            frame=engine.load_data(Path(temp)/"derived/station_047.csv")
            original=pd.read_csv(data/"station_047_test.csv")
            np.testing.assert_allclose(frame.WS.to_numpy(),original.wind_speed_mph*.44704)
            self.assertEqual(str(frame.index[0]),"2025-03-01 00:00:00")
            self.assertEqual(str(frame.index[-1]),"2025-05-31 23:00:00")
            view=canonical_view(data/"station_047_test.csv")
            self.assertEqual(view["Hour (PST)"].iloc[0],2400)
            self.assertEqual(view.Date.iloc[0],"02/28/2025")
            neighbor=load_other_station(Path(temp)/"derived/station_002.csv")
            self.assertEqual(choose_year({47:frame,2:neighbor},["spring"]),2025)

    def test_ridge_ignores_neighbor_features_and_graph_has_target_only_output(self):
        rng=np.random.default_rng(2)
        pack=rng.normal(size=(40,7,7)).astype(np.float32)
        own=pack[:,:,:2].reshape(40,-1)
        y=rng.normal(size=(40,3)).astype(np.float32)
        target_ridge=target_ridge_class(7,2,7)(alpha=.1).fit(pack.reshape(40,-1),y)
        ordinary=Ridge(alpha=.1).fit(own,y)
        np.testing.assert_allclose(target_ridge.predict(pack.reshape(40,-1)),ordinary.predict(own),rtol=1e-6)
        changed=pack.copy();changed[:,:,2:]+=100
        np.testing.assert_allclose(target_ridge.predict(changed.reshape(40,-1)),ordinary.predict(own),rtol=1e-6)
        engine=load_engine("L.1")
        cfg=engine.Config(d_model=8,n_heads=2,kan_hidden=4,kan_grid=3,e_layers=1)
        a=np.array([[.5,.3,.2],[.3,.5,.2],[.2,.3,.5]],dtype=np.float32)
        model=TargetGraphKAN(engine.KANInformer,[2,3,2],a,cfg,torch.device("cpu"))
        x=torch.from_numpy(pack[:2]).requires_grad_()
        out=model(x)
        self.assertEqual(tuple(out.shape),(2,3))
        out.square().mean().backward()
        self.assertGreater(float(x.grad[:,:,2:].abs().sum()),0)

    def test_three_way_synthetic_execution_and_matched_collector(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data,metadata=self.make_data(temp)
            for experiment in ("L","L.1","spatial_L.1"):
                engine=load_engine("L" if experiment=="L" else "L.1")
                original_train=engine.train
                original_model=engine.KANInformer
                def fake_prepare(e,kind,raw,season,year,config,out,label,selection_override=None):
                    cleaned=raw.copy().ffill().bfill()
                    tr,va=int(len(cleaned)*.8),int(len(cleaned)*.9)
                    width=2 if "station_" not in label else 3
                    def fake_bank(frame,selected,k,cfg):
                        x=np.column_stack([np.sin(np.arange(len(frame))/(10+i)) for i in range(width-1)]+[frame.WS.to_numpy()])
                        return x,[f"f{i}" for i in range(width)],{"vmd_relative_absolute_residual":.01}
                    with patch.object(e,"bank_features",side_effect=fake_bank):
                        blocks,scaler,report,labels=e.full_season_windows(cleaned,["WS"],2,config,tr,va)
                    blocks=[tuple(v[:(16 if i==0 else 8)] for v in block) for i,block in enumerate(blocks)]
                    selection={"selected_k":2,"selected_alpha":25.,"method":"synthetic test fixture"}
                    (out/f"{label}_selection.json").write_text(json.dumps(selection))
                    info={"selection":selection,"cleaning":{},"features":labels,"train_end":tr,"validation_end":va,
                          "feature_scaler":scaler_record(scaler),"decomposition":report}
                    return blocks,scaler,config,cleaned,info
                args=argument_parser(experiment).parse_args(["--data",str(data),"--metadata",str(metadata),
                    "--prepared-root",str(root/"prepared"),"--year","2025","--seasons","spring",
                    "--run-id","fixture","--output",str(root/experiment/"results/fixture"),"--epochs","1",
                    "--patience","1","--d-model","8","--n-heads","2","--kan-hidden","4","--kan-grid","3","--neighbors","2"])
                with patch("common.experiment.load_engine",return_value=engine),patch("common.experiment.prepare_station",side_effect=fake_prepare):
                    output=execute(experiment,args)
                self.assertIs(engine.train,original_train)
                self.assertIs(engine.KANInformer,original_model)
                self.assertIs(engine.Ridge,Ridge)
                self.assertTrue((output/"spring_checkpoint.pt").is_file())
                predictions=pd.read_csv(output/"spring_predictions.csv")
                self.assertEqual(set(predictions.station),{47})
            final=summarize("fixture",root)
            self.assertTrue((final/"comparison.md").exists())
            self.assertEqual(set(pd.read_csv(final/"selected_parameters.csv").experiment),{"L","L.1","spatial_L.1"})
            # Matching data/config is insufficient if the actual target rows differ: reject it.
            wrong=root/"spatial_L.1/results/fixture/spring_predictions.csv"
            predictions=pd.read_csv(wrong);predictions.loc[0,"actual"]+=1;predictions.to_csv(wrong,index=False)
            with self.assertRaises(AssertionError): summarize("fixture",root)

    def test_strict_mape_stays_undefined_external_diagnostic_is_labelled(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"spring_predictions.csv"
            pd.DataFrame({"horizon":[1,1],"actual":[0.,2.],"prediction":[1.,3.]}).to_csv(path,index=False)
            result=score_prediction_file(path)[0]
            self.assertIsNone(result["mape"])
            self.assertEqual(result["mape_nonzero"],50.)
            self.assertEqual(result["zero_targets"],1)


if __name__=="__main__": unittest.main()
