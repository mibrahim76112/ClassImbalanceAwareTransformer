import numpy as np
import pandas as pd
import pyreadr
from pathlib import Path

try:
    import cupy as cp
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except Exception:
    cp = np  # fallback
    GPU_AVAILABLE = False

from sklearn.preprocessing import StandardScaler as skStandardScaler


def read_training_data(ff_path: str, ft_path: str):
    """
    Read .RData files using pyreadr. Paths are REQUIRED and validated.
    """
    ff = Path(ff_path)
    ft = Path(ft_path)
    if not ff.exists():
        raise FileNotFoundError(f"Fault-free RData not found: {ff}")
    if not ft.exists():
        raise FileNotFoundError(f"Faulty RData not found: {ft}")

    b1 = pyreadr.read_r(str(ff))['fault_free_training']
    b2 = pyreadr.read_r(str(ft))['faulty_training']
    train_ts = pd.concat([b1, b2]).sort_values(['faultNumber', 'simulationRun'])
    return train_ts


def sample_train_and_test(
    train_ts,
    type_model="supervised",
    post_fault_start=100,
    train_runs=range(1, 25),
    test_runs=range(26, 38),
):
    """
    Slice training/testing windows from the TEP table.
    """
    frames_train, frames_test = [], []
    fault_0 = train_ts[train_ts['faultNumber'] == 0]

    # TRAIN
    if type_model == "supervised":
        for i in sorted(train_ts['faultNumber'].unique()):
            if i == 0:
                frames_train.append(fault_0.iloc[0:42000])
            else:
                b = train_ts[train_ts['faultNumber'] == i]
                per = []
                for x in train_runs:
                    bx = b[b['simulationRun'] == x]
                    per.append(bx.iloc[post_fault_start:500])
                frames_train.append(pd.concat(per))
    else:
        frames_train.append(fault_0)

    sampled_train = pd.concat(frames_train).sort_values(['faultNumber', 'simulationRun', 'sample'])

    # TEST
    for i in sorted(train_ts['faultNumber'].unique()):
        if i == 0:
            frames_test.append(fault_0.iloc[42000:44000])
        else:
            b = train_ts[train_ts['faultNumber'] == i]
            per = []
            for x in test_runs:
                bx = b[b['simulationRun'] == x]
                per.append(bx.iloc[post_fault_start:500])
            frames_test.append(pd.concat(per))

    sampled_test = pd.concat(frames_test).sort_values(['faultNumber', 'simulationRun', 'sample'])
    return sampled_train.reset_index(drop=True), sampled_test.reset_index(drop=True)


def scale_and_window(X_df, scaler, use_gpu=True, y_col='faultNumber',
                     window_size=20, stride=5, return_meta=True):
    """
    Scale + window the multivariate time series into (N_windows, T, F).
    """
    y = X_df[y_col].values
    run_ids = X_df['simulationRun'].values
    samples = X_df['sample'].values
    X = X_df.iloc[:, 3:].values  # features only

    if use_gpu and GPU_AVAILABLE:
        X_scaled = scaler.transform(cp.asarray(X))
        n = max(0, (len(X_scaled) - window_size) // stride + 1)
        X_idx = cp.arange(window_size)[None, :] + cp.arange(n)[:, None] * stride
        y_idx = cp.arange(window_size - 1, window_size - 1 + n * stride, stride)
        X_win = X_scaled[X_idx].get()
        y_win = cp.asarray(y)[y_idx].get()
        run_win = cp.asarray(run_ids)[y_idx].get()
        samp_win = cp.asarray(samples)[y_idx].get()
    else:
        X_scaled = scaler.transform(X)
        n = max(0, (len(X_scaled) - window_size) // stride + 1)
        X_idx = np.arange(window_size)[None, :] + np.arange(n)[:, None] * stride
        X_win = np.stack([X_scaled[idx] for idx in X_idx], axis=0)
        y_idx = np.arange(window_size - 1, window_size - 1 + n * stride, stride)
        y_win = y[y_idx]
        run_win = run_ids[y_idx]
        samp_win = samples[y_idx]

    if return_meta:
        meta = {'faultNumber': y_win, 'simulationRun': run_win, 'sample_end': samp_win}
        return X_win, y_win, meta
    return X_win, y_win


def load_sampled_data(*,
                      window_size=100, stride=5, type_model="supervised", use_gpu=True,
                      ff_path: str, ft_path: str,
                      post_fault_start=100,
                      train_runs=range(1, 25),
                      test_runs=range(26, 38)):
    """
    End-to-end loading + scaling + windowing.
    """
    ts = read_training_data(ff_path, ft_path)
    tr, te = sample_train_and_test(
        ts, type_model,
        post_fault_start=post_fault_start,
        train_runs=train_runs,
        test_runs=test_runs
    )

    fault_free = tr[tr['faultNumber'] == 0].iloc[:, 3:].values
    if use_gpu and GPU_AVAILABLE:
        scaler = cuStandardScaler(); scaler.fit(cp.asarray(fault_free))
    else:
        scaler = skStandardScaler(); scaler.fit(fault_free)

    X_train, y_train, meta_tr = scale_and_window(
        tr, scaler, use_gpu=use_gpu, y_col='faultNumber',
        window_size=window_size, stride=stride, return_meta=True
    )
    X_test, y_test, meta_te = scale_and_window(
        te, scaler, use_gpu=use_gpu, y_col='faultNumber',
        window_size=window_size, stride=stride, return_meta=True
    )
    return (X_train, y_train, meta_tr), (X_test, y_test, meta_te)
