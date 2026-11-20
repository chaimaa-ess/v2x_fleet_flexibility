"""Target-period-safe focused tuning: Random Forest versus CNN-LSTM.

Run from the v2x_forecasting project root. Requires src/target_period_split.py
and existing src/{data_preparation,preprocessing,datasets}.py.
No test predictions, metrics or test-loader iterations are performed.

Examples:
  python run_deep_search.py --models cnn_lstm
  python run_deep_search.py --models random_forest --rf-n-jobs 8
  python run_deep_search.py --models random_forest cnn_lstm
  python run_deep_search.py --models random_forest --budget 16

Default: 16 deterministic, distinct focused configurations per family.
Resume is automatic; results are saved after every completed configuration.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
#import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from torch.utils.data import DataLoader

from src.data_preparation import prepare_forecasting_samples
from src.preprocessing import preprocess_splits, inverse_transform_targets
from src.datasets import create_dataloaders
from src.target_period_split import target_period_split, boundaries_from_raw_index

DATA_PATH = Path('data/residential/polytope_agg_res_df.pkl')
RESULTS_DIR = Path('results/model_deep_search')
CHECKPOINT_DIR = Path('checkpoints/model_deep_search')
TARGETS = ('Pmax', 'Cmax', 'Cmin')
K, L, H, FREQ = 48, 672, 96, '15min'
TRAIN_RATIO, VAL_RATIO = 0.80, 0.05
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 128, 50, 8
SEED, DEFAULT_BUDGET = 42, 16

# Focused search informed by screening. The search space is deliberately
# specified before seeing any corrected-split validation results.
SPACES = {
    'random_forest': {
        'n_estimators': (400, 600, 800),
        'max_depth': (15, 20, 30),
        'max_features': ('sqrt', 0.15, 0.3),
        'min_samples_leaf': (1, 2),
        'min_samples_split': (2, 5),
        'bootstrap': (True,),
    },
    'cnn_lstm': {
        'conv_channels': (64, 96),
        'kernel_size': (3, 5),
        'lstm_hidden': (96, 128, 160),
        'lstm_layers': (1, 2),
        'dropout': (0.0, 0.05, 0.1),
        'head_hidden': (128, 256),
        'learning_rate': (0.0005, 0.001),
        'weight_decay': (0.0, 1e-5, 1e-4),
    },
}
# Include the best screening config for each family as a reference point,
# retrained on the corrected split. This is NOT a reuse of old validation MAE.
ANCHORS = {
    'random_forest': dict(n_estimators=600, max_depth=20, max_features='sqrt',
                          min_samples_leaf=1, min_samples_split=2, bootstrap=True),
    'cnn_lstm': dict(conv_channels=64, kernel_size=3, lstm_hidden=128,
                     lstm_layers=1, dropout=0.0, head_hidden=128,
                     learning_rate=0.001, weight_decay=0.0),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configs_for(model, budget, seed):
    keys = list(SPACES[model])
    pool = [dict(zip(keys, values)) for values in itertools.product(*(SPACES[model][k] for k in keys))]
    anchor = ANCHORS[model]
    pool = [x for x in pool if x != anchor]
    if budget > len(pool) + 1:
        raise ValueError(f'Budget exceeds distinct configurations for {model}')
    rng = random.Random(seed + (1000 if model == 'cnn_lstm' else 0))
    return [anchor] + rng.sample(pool, budget - 1)


def calendar_lookup():
    slots = np.arange(672)
    quarter = slots % 96
    dow = slots // 96
    return np.column_stack((np.sin(2*np.pi*quarter/96), np.cos(2*np.pi*quarter/96),
                            np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7))).astype(np.float32)


def tree_matrix(split):
    hist = np.asarray(split['X_history'], dtype=np.float32)
    origin = pd.DatetimeIndex(pd.to_datetime(split['origin']))
    slots = origin.dayofweek.to_numpy()*96 + origin.hour.to_numpy()*4 + origin.minute.to_numpy()//15
    offsets = np.arange(1-hist.shape[1], 1, dtype=np.int32)
    # Build in batches to avoid simultaneously allocating huge 3D calendar arrays.
    X = np.empty((len(hist), hist.shape[1]*7), dtype=np.float32)
    lookup = calendar_lookup()
    for start in range(0, len(hist), 256):
        end = min(start+256, len(hist))
        cal = lookup[(slots[start:end, None] + offsets[None, :]) % 672]
        X[start:end] = np.concatenate((hist[start:end], cal), axis=-1).reshape(end-start, -1)
    Y = np.ascontiguousarray(split['Y'].reshape(len(hist), -1), dtype=np.float32)
    return X, Y


def metrics(true, pred):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    e = pred - true
    yt, yp = true.ravel(), pred.ravel()
    denom = np.square(yt - yt.mean()).sum()
    out = dict(val_mae=float(np.abs(e).mean()), val_rmse=float(np.sqrt(np.square(e).mean())),
               val_r2=float(1 - np.square(yt-yp).sum()/denom) if denom else float('nan'))
    for j, name in enumerate(TARGETS):
        t, p = true[..., j], pred[..., j]
        den = np.square(t-t.mean()).sum()
        out[name+'_mae'] = float(np.abs(p-t).mean())
        out[name+'_rmse'] = float(np.sqrt(np.square(p-t).mean()))
        out[name+'_r2'] = float(1 - np.square(p-t).sum()/den) if den else float('nan')
    return out


class CNNLSTM(nn.Module):
    """Same architecture as the screening runner (not a changed model family)."""
    def __init__(self, conv_channels, kernel_size, lstm_hidden, lstm_layers,
                 dropout, head_hidden):
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


def batch_to_device(batch, device):
    return (batch['history'].to(device, dtype=torch.float32),
            batch['history_time'].to(device, dtype=torch.float32),
            batch['target'].to(device, dtype=torch.float32))


def train_cnn(config, train_loader, val_loader, device, seed, checkpoint):
    set_seed(seed)
    arch = {k: v for k, v in config.items() if k not in ('learning_rate', 'weight_decay')}
    model = CNNLSTM(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, no_improve = float('inf'), 0, 0
    start = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        for batch in train_loader:
            x, cal, y = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x, cal), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, cal, y = batch_to_device(batch, device)
                loss = loss_fn(model(x, cal), y)
                total += float(loss.item())*len(x)
                count += len(x)
        score = total/count
        print(f'  epoch {epoch:02d}: scaled validation MSE={score:.6f}', flush=True)
        if score < best_loss:
            best_loss, best_epoch, no_improve = score, epoch, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x, cal, y = batch_to_device(batch, device)
            preds.append(model(x, cal).cpu().numpy())
            targets.append(y.cpu().numpy())
    pred, true = np.concatenate(preds), np.concatenate(targets)
    result = dict(best_epoch=best_epoch, epochs_trained=epoch,
                  best_val_loss_scaled=best_loss,
                  training_seconds=time.perf_counter()-start,
                  n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return pred, true, result


def atomic_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def manifest_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', choices=('random_forest', 'cnn_lstm'),
                   default=['random_forest', 'cnn_lstm'])
    p.add_argument('--budget', type=int, default=DEFAULT_BUDGET)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--rf-n-jobs', type=int, default=-1)
    p.add_argument('--output-dir', type=Path, default=RESULTS_DIR)
    args = p.parse_args()
    if args.budget < 1:
        p.error('--budget must be >= 1')
    set_seed(args.seed)
    df = pd.read_pickle(DATA_PATH).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('Data index must be DatetimeIndex')
    boundaries_path = args.output_dir/'boundaries.json'
    proposed = boundaries_from_raw_index(df, TRAIN_RATIO, VAL_RATIO)
    proposed = {'train_end': str(proposed[0]), 'val_end': str(proposed[1]),
                'raw_first': str(df.index.min()), 'raw_last': str(df.index.max())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if boundaries_path.exists():
        saved = json.loads(boundaries_path.read_text())
        if saved != proposed:
            raise RuntimeError('Existing split boundaries differ. Refusing to overwrite.')
    else:
        boundaries_path.write_text(json.dumps(proposed, indent=2))
    print('Fixed raw-time boundaries:', proposed, flush=True)
    samples = prepare_forecasting_samples(df=df, k=K, history_steps=L, horizon=H,
                                           freq=FREQ, target_cols=TARGETS,
                                           feature_cols=TARGETS, verbose=True)
    splits = target_period_split(samples, train_end=proposed['train_end'],
                                 val_end=proposed['val_end'], verbose=True)
    scaled, scalers = preprocess_splits(splits)
    # The test split is reserved, never passed to a model or metric.
    del splits, samples
    sampled = {m: configs_for(m, args.budget, args.seed) for m in args.models}
    # Each model has its own manifest, so running --models separately is safe.
    for model, configs in sampled.items():
        path = args.output_dir/f'{model}_manifest.json'
        record = dict(model=model, seed=args.seed, budget=args.budget,
                      scenario=dict(k=K, L=L, H=H, freq=FREQ, calendar='historical'),
                      split=proposed, configs=configs, selection='original-unit validation MAE',
                      test_evaluated=False)
        if path.exists():
            if manifest_hash(json.loads(path.read_text())) != manifest_hash(record):
                raise RuntimeError(f'{path} differs; refusing to overwrite or mix runs')
        else:
            path.write_text(json.dumps(record, indent=2))
    loaders = None
    if 'cnn_lstm' in args.models:
        # create_dataloaders expects train/val/test, but we NEVER iterate test.
        loaders = create_dataloaders(scaled, batch_size=BATCH_SIZE,
                                     num_workers=0, seed=args.seed, calendar_mode='history')
        check = next(iter(loaders['train']))
        assert check['history'].shape[-2:] == (L, 3)
        assert check['history_time'].shape[-2:] == (L, 4)
        del check
    tree_arrays = None
    if 'random_forest' in args.models:
        print('Materializing RF train/validation matrices...', flush=True)
        xt, yt = tree_matrix(scaled['train'])
        xv, yv = tree_matrix(scaled['val'])
        tree_arrays = xt, yt, xv, yv
        print(f'RF matrices: train {xt.shape}, val {xv.shape}', flush=True)
    device = torch.device(args.device)
    if 'cnn_lstm' in args.models and device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    for model in args.models:
        result_path = args.output_dir/f'{model}_results.csv'
        rows = pd.read_csv(result_path).to_dict('records') if result_path.exists() else []
        completed = {int(r['config_id']) for r in rows if r.get('status') == 'ok'}
        for config_id, config in enumerate(sampled[model], 1):
            if config_id in completed:
                print(f'SKIP {model} config {config_id}: completed', flush=True)
                continue
            print(f'\n{model} config {config_id}/{args.budget}: {json.dumps(config)}', flush=True)
            seed = args.seed  # same initialization seed protocol as screening
            try:
                if model == 'random_forest':
                    xt, yt, xv, yv = tree_arrays
                    tic = time.perf_counter()
                    rf = RandomForestRegressor(random_state=seed, n_jobs=args.rf_n_jobs,
                                               **config)
                    rf.fit(xt, yt)
                    ps = rf.predict(xv).reshape(-1, H, 3).astype(np.float32)
                    ts = yv.reshape(-1, H, 3)
                    extra = dict(training_seconds=time.perf_counter()-tic,
                                 n_parameters=np.nan, best_epoch=np.nan,
                                 epochs_trained=np.nan, best_val_loss_scaled=np.nan)
                    del rf
                else:
                    ckpt = CHECKPOINT_DIR/model/f'config_{config_id:02d}.pt'
                    ps, ts, extra = train_cnn(config, loaders['train'], loaders['val'],
                                              device, seed, ckpt)
                pred = inverse_transform_targets(ps, scalers.target_scaler)
                true = inverse_transform_targets(ts, scalers.target_scaler)
                row = dict(model=model, config_id=config_id, status='ok', seed=seed,
                           k=K, history_steps=L, horizon=H,
                           train_samples=len(scaled['train']['Y']),
                           val_samples=len(scaled['val']['Y']),
                           test_samples_reserved=len(scaled['test']['Y']),
                           hyperparameters=json.dumps(config, sort_keys=True),
                           **extra, **metrics(true, pred))
                print(f"Validation MAE: {row['val_mae']:.4f}", flush=True)
            except Exception as exc:
                row = dict(model=model, config_id=config_id, status='failed', seed=seed,
                           hyperparameters=json.dumps(config, sort_keys=True),
                           error=f'{type(exc).__name__}: {exc}')
                print(f'FAILED: {row["error"]}', flush=True)
            rows = [r for r in rows if int(r['config_id']) != config_id] + [row]
            rows.sort(key=lambda r: int(r['config_id']))
            atomic_csv(rows, result_path)
            gc.collect()
        good = pd.DataFrame([r for r in rows if r.get('status') == 'ok'])
        if not good.empty:
            good = good.sort_values('val_mae')
            print(f'\n{model} best configurations:\n'+good[['config_id','val_mae','val_rmse','training_seconds']].head(5).to_string(index=False))
            print(f"Median MAE={good.val_mae.median():.4f}; IQR={good.val_mae.quantile(.75)-good.val_mae.quantile(.25):.4f}")
    print('\nDone. Test was NOT evaluated. Previous screening outputs were not changed.')


if __name__ == '__main__':
    main()
