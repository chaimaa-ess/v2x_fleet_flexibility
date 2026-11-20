"""Calendar ablation with fixed 7-day history and the window-search TCN."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.data_preparation import prepare_forecasting_samples
from src.preprocessing import chronological_split, inverse_transform_targets, preprocess_splits
from src.datasets import create_dataloaders
from src.training import TrainingConfig, count_trainable_parameters, predict, train_model

DATA_PATH=Path("data/residential/polytope_agg_res_df.pkl")
DATASET_NAME="residential"; HISTORY_STEPS=672; LEADS=(1,4,48); HORIZON=96; FREQ="15min"
TARGET_COLS=("Pmax","Cmax","Cmin")
RESULTS_DIR=Path("results/calendar_ablation"); CHECKPOINT_DIR=Path("checkpoints/calendar_ablation")
BATCH_SIZE=128; MAX_EPOCHS=50; PATIENCE=8; LEARNING_RATE=1e-3; WEIGHT_DECAY=1e-4; SEED=42
TRAIN_RATIO=.80; VAL_RATIO=.05; TEST_RATIO=.15
DILATIONS=(1,2,4,8,16,32,64,128,256)
ABLATIONS={
 #"A":dict(description="history only",calendar_mode="none",use_history_calendar=False,use_future_calendar=False),
 "B":dict(description="history + historical calendar",calendar_mode="history",use_history_calendar=True,use_future_calendar=False),
 #"C":dict(description="history + historical + future calendar",calendar_mode="history_future",use_history_calendar=True,use_future_calendar=True),
}

class Chomp1d(nn.Module):
    def __init__(self,n:int): super().__init__(); self.n=n
    def forward(self,x): return x if self.n==0 else x[:,:,:-self.n].contiguous()

class ResidualCausalBlock(nn.Module):
    def __init__(self,channels:int,kernel_size:int,dilation:int,dropout:float):
        super().__init__(); pad=(kernel_size-1)*dilation
        self.conv=nn.Conv1d(channels,channels,kernel_size,dilation=dilation,padding=pad)
        self.chomp=Chomp1d(pad); self.norm=nn.BatchNorm1d(channels)
        self.act=nn.ReLU(); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        z=self.drop(self.act(self.norm(self.chomp(self.conv(x)))))
        return self.act(z+x)

class CalendarAblationTCN(nn.Module):
    """Same encoder/head pattern as WindowSearchTCN; only calendar inputs vary."""
    def __init__(self,n_features=3,n_targets=3,horizon=96,channels=32,kernel_size=3,
                 dropout=.1,dilations:DILATIONS.__class__=DILATIONS,head_hidden=128,
                 calendar_dim=4,use_history_calendar=False,use_future_calendar=False):
        super().__init__()
        if use_future_calendar and not use_history_calendar:
            raise ValueError("Future calendar requires historical calendar.")
        self.n_features=n_features; self.n_targets=n_targets; self.horizon=horizon
        self.calendar_dim=calendar_dim; self.use_history_calendar=use_history_calendar
        self.use_future_calendar=use_future_calendar
        enc=n_features+(calendar_dim if use_history_calendar else 0)
        self.input_projection=nn.Conv1d(enc,channels,1)
        self.blocks=nn.Sequential(*[ResidualCausalBlock(channels,kernel_size,d,dropout) for d in dilations])
        head_in=channels+(horizon*calendar_dim if use_future_calendar else 0)
        self.head=nn.Sequential(nn.Linear(head_in,head_hidden),nn.ReLU(),nn.Dropout(dropout),
                                nn.Linear(head_hidden,horizon*n_targets))
    def forward(self,history,history_time=None,future_time=None):
        if history.ndim!=3 or history.shape[-1]!=self.n_features:
            raise ValueError(f"Bad history shape: {tuple(history.shape)}")
        if self.use_history_calendar:
            if history_time is None: raise ValueError("history_time is required")
            if history_time.shape[:2]!=history.shape[:2] or history_time.shape[-1]!=self.calendar_dim:
                raise ValueError(f"Bad history_time shape: {tuple(history_time.shape)}")
            x=torch.cat([history,history_time],dim=-1)
        else: x=history
        x=self.blocks(self.input_projection(x.transpose(1,2)))[:,:,-1]
        if self.use_future_calendar:
            if future_time is None: raise ValueError("future_time is required")
            expected=(history.shape[0],self.horizon,self.calendar_dim)
            if tuple(future_time.shape)!=expected: raise ValueError(f"Expected future_time {expected}, got {tuple(future_time.shape)}")
            x=torch.cat([x,future_time.flatten(start_dim=1)],dim=1)
        return self.head(x).view(-1,self.horizon,self.n_targets)

def metrics(y_true,y_pred):
    a=np.asarray(y_true,dtype=float); b=np.asarray(y_pred,dtype=float); e=b-a
    flat=a.reshape(-1); pred=b.reshape(-1); ss=np.sum((flat-flat.mean())**2)
    return dict(val_mae=float(np.mean(np.abs(e))),val_rmse=float(np.sqrt(np.mean(e**2))),
                val_r2=float('nan') if ss==0 else float(1-np.sum((flat-pred)**2)/ss))

def target_metrics(y_true,y_pred):
    out={}
    for j,n in enumerate(TARGET_COLS):
        a=y_true[...,j]; b=y_pred[...,j]; e=b-a; ss=np.sum((a.reshape(-1)-a.mean())**2)
        out[f"{n}_mae"]=float(np.mean(np.abs(e))); out[f"{n}_rmse"]=float(np.sqrt(np.mean(e**2)))
        out[f"{n}_r2"]=float('nan') if ss==0 else float(1-np.sum((a-b)**2)/ss)
    return out

def load_df():
    if not DATA_PATH.exists(): raise FileNotFoundError(DATA_PATH.resolve())
    df=pd.read_pickle(DATA_PATH)
    if not isinstance(df.index,pd.DatetimeIndex): raise TypeError("DataFrame index must be DatetimeIndex")
    missing=[c for c in TARGET_COLS if c not in df.columns]
    if missing: raise KeyError(f"Missing columns: {missing}")
    return df.sort_index()

def prepare_k(df,k):
    print(f"\n{'='*78}\nPREPARE k={k} | L={HISTORY_STEPS}\n{'='*78}")
    s=prepare_forecasting_samples(df=df,k=k,history_steps=HISTORY_STEPS,horizon=HORIZON,freq=FREQ,
                                  target_cols=TARGET_COLS,feature_cols=TARGET_COLS,verbose=True)
    d=s["diagnostics"]
    if d["valid_samples"]<20: raise ValueError(f"Only {d['valid_samples']} valid samples")
    splits=chronological_split(s,train_ratio=TRAIN_RATIO,val_ratio=VAL_RATIO,test_ratio=TEST_RATIO,verbose=True)
    scaled,scalers=preprocess_splits(splits)
    return scaled,scalers,d

def run_config(k,name,cfg,scaled,scalers,diag,device):
    print(f"\n{'-'*78}\n{name}: {cfg['description']} | k={k}\n{'-'*78}")
    loaders=create_dataloaders(scaled,batch_size=BATCH_SIZE,num_workers=0,seed=SEED,
                               calendar_mode=cfg["calendar_mode"])
    model=CalendarAblationTCN(n_features=3,n_targets=3,horizon=HORIZON,
                              use_history_calendar=cfg["use_history_calendar"],
                              use_future_calendar=cfg["use_future_calendar"])
    nparams=count_trainable_parameters(model)
    tc=TrainingConfig(max_epochs=MAX_EPOCHS,learning_rate=LEARNING_RATE,weight_decay=WEIGHT_DECAY,
                      patience=PATIENCE,min_delta=0.0,gradient_clip_norm=1.0,device=device,seed=SEED,verbose=True)
    ckpt=CHECKPOINT_DIR/DATASET_NAME/f"k{k}_config_{name}.pt"
    tr=train_model(model,loaders["train"],loaders["val"],tc,criterion=nn.MSELoss(),checkpoint_path=ckpt)
    vo=predict(model,loaders["val"],device=device,return_targets=True)
    yp=inverse_transform_targets(vo["predictions"],scalers.target_scaler)
    yt=inverse_transform_targets(vo["targets"],scalers.target_scaler)
    row={"dataset":DATASET_NAME,"k":k,"history_steps":HISTORY_STEPS,"calendar_config":name,
         "calendar_description":cfg["description"],"calendar_mode":cfg["calendar_mode"],
         "use_history_calendar":cfg["use_history_calendar"],"use_future_calendar":cfg["use_future_calendar"],
         "valid_samples":diag["valid_samples"],"train_samples":len(scaled["train"]["Y"]),
         "val_samples":len(scaled["val"]["Y"]),"test_samples_reserved":len(scaled["test"]["Y"]),
         "n_parameters":nparams,"best_epoch":tr.best_epoch,"best_val_loss_scaled":tr.best_val_loss,
         "epochs_trained":tr.epochs_trained,"training_seconds":tr.training_seconds,**metrics(yt,yp),**target_metrics(yt,yp)}
    print(f"Validation | {name} | k={k} | MAE={row['val_mae']:.4f} | RMSE={row['val_rmse']:.4f} | R2={row['val_r2']:.6f}")
    return row

def save(rows):
    RESULTS_DIR.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows); df.to_csv(RESULTS_DIR/"calendar_ablation_results.csv",index=False)
    if "val_mae" not in df: return pd.DataFrame()
    ok=df[df.val_mae.notna()].copy()
    if ok.empty: return ok
    ok["mae_rank"]=ok.groupby("k")["val_mae"].rank(method="min")
    ok=ok.sort_values(["k","mae_rank","calendar_config"]); ok.to_csv(RESULTS_DIR/"calendar_ablation_ranked.csv",index=False)
    cols=["k","calendar_config","calendar_description","n_parameters","val_mae","val_rmse","val_r2","mae_rank"]
    ok[cols].to_csv(RESULTS_DIR/"calendar_ablation_summary.csv",index=False)
    return ok

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default=None); p.add_argument("--k",type=int,nargs="*",default=list(LEADS))
    p.add_argument("--configs",nargs="*",choices=tuple(ABLATIONS),default=list(ABLATIONS)); a=p.parse_args()
    RESULTS_DIR.mkdir(parents=True,exist_ok=True); CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True)
    df=load_df()
    record={"dataset":DATASET_NAME,"history_steps":HISTORY_STEPS,"leads":a.k,"configs":{x:ABLATIONS[x] for x in a.configs},
            "horizon":HORIZON,"batch_size":BATCH_SIZE,"max_epochs":MAX_EPOCHS,"patience":PATIENCE,
            "learning_rate":LEARNING_RATE,"weight_decay":WEIGHT_DECAY,"seed":SEED,"test_evaluated":False}
    (RESULTS_DIR/"calendar_ablation_config.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    rows=[]
    for k in a.k:
        scaled,scalers,diag=prepare_k(df,k)  # ONCE per k; reused for A/B/C
        for name in a.configs:
            try:
                rows.append(run_config(k,name,ABLATIONS[name],scaled,scalers,diag,a.device)); save(rows)
            except Exception as e:
                print(f"FAILED | k={k} | config={name} | {type(e).__name__}: {e}")
    good=[r for r in rows if "val_mae" in r and np.isfinite(r["val_mae"])]
    if not good: raise RuntimeError("No calendar-ablation experiment completed successfully")
    ranked=save(good)
    print("\nCALENDAR ABLATION COMPLETE")
    print(ranked[["k","calendar_config","calendar_description","n_parameters","val_mae","val_rmse","val_r2","mae_rank"]].to_string(index=False))
    print("\nTest partition was reserved and NOT evaluated.")

if __name__=="__main__": main()
