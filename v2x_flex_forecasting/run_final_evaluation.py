"""Final EV flexibility forecasts: residential/commercial x k={1,4,48} x RF/CNN-LSTM/Ridge.

Run from project root with existing src/data_preparation.py, src/preprocessing.py,
src/target_period_split.py. All targets and predictions are saved in original units.
No hyperparameter tuning or test-driven selection occurs here.

Examples:
  python run_final_evaluation.py
  python run_final_evaluation.py --datasets residential --leads 1 4 --models ridge random_forest
  python run_final_evaluation.py --datasets commercial --models cnn_lstm --device cuda
  python run_final_evaluation.py --datasets commercial --models random_forest --rf-n-jobs 8
  python run_final_evaluation.py --force --datasets residential --leads 48 --models ridge

Resume: each finished task is skipped only if its output manifest matches the current
configuration and required prediction/model files exist. Incomplete tasks restart.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.data_preparation import prepare_forecasting_samples
from src.preprocessing import preprocess_splits, inverse_transform_targets
from src.target_period_split import target_period_split, boundaries_from_raw_index

DATA = {
    'residential': Path('data/residential/polytope_agg_res_df.pkl'),
    'commercial': Path('data/commercial/polytope_agg_com_df.pkl'),
}
TARGETS = ('Pmax', 'Cmax', 'Cmin')
L, H, FREQ, SEED = 672, 96, '15min', 42
TRAIN_RATIO, VAL_RATIO = 0.80, 0.05
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 128, 50, 8
RF_CONFIG = dict(n_estimators=600, max_depth=20, max_features='sqrt',
                 min_samples_leaf=1, min_samples_split=2, bootstrap=True)
CNN_CONFIG = dict(conv_channels=64, kernel_size=5, lstm_hidden=128,
                  lstm_layers=1, dropout=0.05, head_hidden=256,
                  learning_rate=5e-4, weight_decay=1e-5)
RIDGE_ALPHA = 1000.0


def stable_json(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def digest(obj):
    return hashlib.sha256(stable_json(obj).encode()).hexdigest()


def atomic_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calendar_lookup():
    slots = np.arange(672)
    quarter, dow = slots % 96, slots // 96
    return np.column_stack((np.sin(2*np.pi*quarter/96), np.cos(2*np.pi*quarter/96),
                            np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7))).astype(np.float32)

LOOKUP = calendar_lookup()
OFFSETS = np.arange(1-L, 1, dtype=np.int32)


def origin_slots(origin):
    ts = pd.DatetimeIndex(pd.to_datetime(origin))
    return (ts.dayofweek.to_numpy()*96 + ts.hour.to_numpy()*4 +
            ts.minute.to_numpy()//15).astype(np.int32)


def tree_matrix(split):
    hist = np.asarray(split['X_history'], dtype=np.float32)
    slots = origin_slots(split['origin'])
    n = len(hist)
    X = np.empty((n, L*7), dtype=np.float32)
    for a in range(0, n, 256):
        b = min(a+256, n)
        cal = LOOKUP[(slots[a:b, None] + OFFSETS[None, :]) % 672]
        X[a:b] = np.concatenate((hist[a:b], cal), axis=-1).reshape(b-a, -1)
    return X


class HistoricalDataset(Dataset):
    def __init__(self, split):
        self.x = np.asarray(split['X_history'], dtype=np.float32)
        self.y = np.asarray(split['Y'], dtype=np.float32)
        self.slots = origin_slots(split['origin'])
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        cal = LOOKUP[(self.slots[i] + OFFSETS) % 672]
        return torch.from_numpy(self.x[i]), torch.from_numpy(cal), torch.from_numpy(self.y[i])


class CNNLSTM(nn.Module):
    """Exactly the architecture defined in the uploaded deep-search runner."""
    def __init__(self, conv_channels, kernel_size, lstm_hidden, lstm_layers, dropout, head_hidden):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(7, conv_channels, kernel_size,
                                          padding=kernel_size//2), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(lstm_hidden, head_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head_hidden, H*3))
    def forward(self, history, history_time):
        x = torch.cat((history, history_time), dim=-1)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.head(x[:, -1]).view(-1, H, 3)


def predict_cnn(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, cal, _ in loader:
            out.append(model(x.to(device), cal.to(device)).cpu().numpy())
    return np.concatenate(out)


def train_cnn(scaled, model_dir, device):
    seed_all(SEED)
    arch = {k:v for k,v in CNN_CONFIG.items() if k not in ('learning_rate','weight_decay')}
    model = CNNLSTM(**arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CNN_CONFIG['learning_rate'],
                                   weight_decay=CNN_CONFIG['weight_decay'])
    train_loader = DataLoader(HistoricalDataset(scaled['train']), batch_size=BATCH_SIZE,
                              shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(HistoricalDataset(scaled['val']), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(HistoricalDataset(scaled['test']), batch_size=BATCH_SIZE, shuffle=False)
    best_loss, best_epoch, bad = float('inf'), 0, 0
    checkpoint = model_dir / 'model.pt'
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        for x, cal, y in train_loader:
            x, cal, y = x.to(device), cal.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x, cal), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite CNN-LSTM training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        val_sum, n = 0., 0
        with torch.no_grad():
            for x, cal, y in val_loader:
                x, cal, y = x.to(device), cal.to(device), y.to(device)
                val_sum += nn.functional.mse_loss(model(x, cal), y).item() * len(x)
                n += len(x)
        score = val_sum / n
        print(f'  epoch {epoch:02d} val scaled MSE={score:.6f}', flush=True)
        if score < best_loss:
            best_loss, best_epoch, bad = score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    predictions = {'val': predict_cnn(model, val_loader, device),
                   'test': predict_cnn(model, test_loader, device)}
    return predictions, dict(best_epoch=best_epoch, epochs_trained=epoch,
                             best_val_loss_scaled=best_loss)


def fit_predict(model_name, scaled, model_dir, device, rf_n_jobs):
    if model_name == 'cnn_lstm':
        return train_cnn(scaled, model_dir, device)
    X_train = tree_matrix(scaled['train'])
    Y_train = np.asarray(scaled['train']['Y'], dtype=np.float32).reshape(len(X_train), -1)
    if model_name == 'random_forest':
        model = RandomForestRegressor(random_state=SEED, n_jobs=rf_n_jobs, **RF_CONFIG)
    elif model_name == 'ridge':
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver='cholesky')
    else:
        raise ValueError(model_name)
    model.fit(X_train, Y_train)
    del X_train, Y_train
    joblib.dump(model, model_dir/'model.joblib', compress=3)
    predictions = {}
    for part in ('val', 'test'):
        X = tree_matrix(scaled[part])
        predictions[part] = model.predict(X).reshape(-1, H, 3).astype(np.float32)
        del X
    return predictions, {}


def metric_dict(y, p):
    e = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    yt, yp = np.asarray(y, dtype=np.float64).ravel(), np.asarray(p, dtype=np.float64).ravel()
    den = np.square(yt-yt.mean()).sum()
    result = {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt(np.square(e).mean())),
              'r2': float(1-np.square(yt-yp).sum()/den) if den else None}
    for j, name in enumerate(TARGETS):
        t, p0 = np.asarray(y, dtype=np.float64)[...,j], np.asarray(p, dtype=np.float64)[...,j]
        den = np.square(t-t.mean()).sum()
        result[name+'_mae'] = float(np.abs(p0-t).mean())
        result[name+'_rmse'] = float(np.sqrt(np.square(p0-t).mean()))
        result[name+'_bias'] = float((p0-t).mean())
        result[name+'_r2'] = float(1-np.square(p0-t).sum()/den) if den else None
    return result


def data_signature(path, df):
    st = path.stat()
    return {'path': str(path.resolve()), 'size_bytes': st.st_size,
            'mtime_ns': st.st_mtime_ns, 'first': str(df.index.min()),
            'last': str(df.index.max()), 'rows': len(df)}


def get_boundaries(dataset, df, root):
    # Reuse residential deep-search boundaries if available to keep k=48 directly comparable.
    legacy = Path('results/model_deep_search/boundaries.json') if dataset == 'residential' else None
    path = root/dataset/'boundaries.json'
    if path.exists():
        b = json.loads(path.read_text())
    elif legacy is not None and legacy.exists():
        b = json.loads(legacy.read_text())
        atomic_json(b, path)
    else:
        a, v = boundaries_from_raw_index(df, TRAIN_RATIO, VAL_RATIO)
        b = {'train_end': str(a), 'val_end': str(v),
             'raw_first': str(df.index.min()), 'raw_last': str(df.index.max())}
        atomic_json(b, path)
    if b['raw_first'] != str(df.index.min()) or b['raw_last'] != str(df.index.max()):
        raise RuntimeError(f'{path}: raw time range differs from saved split')
    return b


def is_complete(path, expected):
    manifest = path/'manifest.json'
    if not manifest.exists():
        return False
    saved = json.loads(manifest.read_text())
    if saved.get('signature') != digest(expected):
        raise RuntimeError(f'Existing output differs from current experiment: {path}. '
                           'Use a different --output-dir or --force to overwrite intentionally.')
    if saved.get('status') != 'complete':
        return False
    return all((path/f).exists() for f in ('predictions_val.npz','predictions_test.npz',
                                           'metrics.json','scalers.joblib',
                                           'model.pt' if expected['model']=='cnn_lstm' else 'model.joblib'))


def run_task(dataset, k, model, scaled, scalers, b, dsig, args):
    path = args.output_dir/dataset/f'k{k:02d}'/model
    path.mkdir(parents=True, exist_ok=True)
    cfg = {'random_forest': RF_CONFIG, 'cnn_lstm': CNN_CONFIG, 'ridge': {'alpha': RIDGE_ALPHA}}[model]
    expected = {'dataset': dataset, 'lead': k, 'model': model, 'config': cfg,
                'L': L, 'H': H, 'frequency': FREQ, 'calendar': 'history',
                'seed': SEED, 'train_ratio': TRAIN_RATIO, 'val_ratio': VAL_RATIO,
                'boundaries': b, 'data': dsig,
                'cnn_protocol': {'batch_size':BATCH_SIZE,'max_epochs':MAX_EPOCHS,'patience':PATIENCE,
                                 'loss':'scaled MSE','optimizer':'AdamW','gradient_clip':1.0} if model=='cnn_lstm' else None}
    if not args.force and is_complete(path, expected):
        print(f'SKIP completed: {dataset} k={k} {model}', flush=True)
        return
    atomic_json({'status':'running','signature':digest(expected),**expected}, path/'manifest.json')
    started = time.perf_counter()
    predictions, extra = fit_predict(model, scaled, path, torch.device(args.device), args.rf_n_jobs)
    results = {}
    for part in ('val','test'):
        pred = inverse_transform_targets(predictions[part], scalers.target_scaler).astype(np.float32)
        true = inverse_transform_targets(scaled[part]['Y'], scalers.target_scaler).astype(np.float32)
        if pred.shape != true.shape or not np.isfinite(pred).all():
            raise ValueError(f'Invalid {part} predictions: {pred.shape}, {true.shape}')
        origin = np.asarray(scaled[part]['origin'], dtype='datetime64[ns]')
        target_start = np.asarray(scaled[part]['target_start'], dtype='datetime64[ns]')
        atomic_npz(path/f'predictions_{part}.npz', y_true=true, y_pred=pred,
                   origin=origin, target_start=target_start,
                   target_names=np.asarray(TARGETS), k=np.asarray(k),
                   freq=np.asarray(FREQ))
        results[part] = {'n_samples':len(true), **metric_dict(true, pred)}
        print(f'  {part}: N={len(true)} MAE={results[part]["mae"]:.4f}', flush=True)
        del pred, true
    joblib.dump(scalers, path/'scalers.joblib', compress=3)
    results['training_and_prediction_seconds'] = time.perf_counter()-started
    results['training_info'] = extra
    atomic_json(results, path/'metrics.json')
    atomic_json({'status':'complete','signature':digest(expected),**expected,
                 'training_info':extra}, path/'manifest.json')
    print(f'COMPLETE {dataset} k={k} {model}', flush=True)


def summarize(root):
    rows = []
    for manifest in sorted(root.glob('*/k??/*/manifest.json')):
        m = json.loads(manifest.read_text())
        if m.get('status') != 'complete':
            continue
        met = json.loads((manifest.parent/'metrics.json').read_text())
        for part in ('val','test'):
            rows.append({'dataset':m['dataset'],'k':m['lead'],'model':m['model'],
                         'partition':part,**met[part]})
    if rows:
        df = pd.DataFrame(rows)
        tmp = root/'metrics_summary.tmp.csv'
        df.to_csv(tmp,index=False)
        os.replace(tmp,root/'metrics_summary.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets',nargs='+',choices=tuple(DATA),default=list(DATA))
    ap.add_argument('--leads',nargs='+',type=int,choices=(1,4,48),default=[1,4,48])
    ap.add_argument('--models',nargs='+',choices=('random_forest','cnn_lstm','ridge'),
                    default=['ridge','cnn_lstm','random_forest'])
    ap.add_argument('--output-dir',type=Path,default=Path('results/final_forecasts'))
    ap.add_argument('--rf-n-jobs',type=int,default=-1)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--force',action='store_true',help='Intentionally overwrite selected tasks')
    args = ap.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available() and 'cnn_lstm' in args.models:
        ap.error('CUDA requested but PyTorch CUDA unavailable')
    seed_all(SEED)
    for dataset in args.datasets:
        data_path = DATA[dataset]
        if not data_path.exists():
            raise FileNotFoundError(f'Missing dataset: {data_path}')
        df = pd.read_pickle(data_path).sort_index()
        if not isinstance(df.index,pd.DatetimeIndex) or df.index.has_duplicates:
            raise ValueError(f'{dataset}: expected unique DatetimeIndex')
        b = get_boundaries(dataset,df,args.output_dir)
        dsig = data_signature(data_path,df)
        print(f'\nDATASET {dataset} | rows={len(df)} | boundaries={b}',flush=True)
        for k in args.leads:
            print(f'\nLEAD k={k} | constructing valid windows',flush=True)
            samples = prepare_forecasting_samples(df=df,k=k,history_steps=L,horizon=H,
                freq=FREQ,target_cols=TARGETS,feature_cols=TARGETS,verbose=True)
            try:
                splits = target_period_split(samples,train_end=b['train_end'],
                                              val_end=b['val_end'],verbose=True)
            except ValueError as exc:
                raise RuntimeError(f'{dataset} k={k}: insufficient complete windows in a '
                                   f'partition. Inspect commercial gaps and fixed dates. {exc}') from exc
            del samples
            scaled, scalers = preprocess_splits(splits)
            del splits
            for model in args.models:
                print(f'\nRUN {dataset} k={k} {model}',flush=True)
                run_task(dataset,k,model,scaled,scalers,b,dsig,args)
                summarize(args.output_dir)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del scaled,scalers
            gc.collect()
        del df
    summarize(args.output_dir)
    print('\nAll requested tasks complete. Predictions saved in original units.',flush=True)


if __name__=='__main__':
    main()
