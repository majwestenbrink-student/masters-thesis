import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import gc
import random
import time
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ============================================
# PARAMETERS
# ============================================

FILE_PATH       = r"\\GIMECB01\HOMEDIR-VZ$\westenb\Thesis\Data\Forecast_Ready.csv"
CKPT_DIR        = r"\\GIMECB01\HOMEDIR-VZ$\westenb\Thesis\Data\checkpoints"

N_LAGS       = 5
LAG_STEP     = 5
N_FACTORS    = 4
TRAIN_WINDOW = 1000
FUTURE       = 21
LSTM_UNITS   = 20
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 0.001
PATIENCE     = 3
TARGET_PC    = 3        # change to 1, 2, 3, 4

N_WORKERS    = 12        # safe for most machines

STATIC_FEATURES = ['EIGENVAL_1', 'EIGENVAL_2', 'EIGENVAL_3', 'EIGENVAL_4']
TARGET_COL      = f'FUTURE_PC{TARGET_PC}_DIRECTION'
PRED_COL        = f'FUTURE_PC{TARGET_PC}_PREDICTION'

LAG_VALUES     = [k * LAG_STEP for k in range(N_LAGS - 1, -1, -1)]  # [20,15,10,5,0]
MAX_LAG        = max(LAG_VALUES)       # 20
N_SEQ_FEATURES = N_FACTORS + 1 + 1    # 4 PCs + RMSE + IV = 6

# ============================================
# NUMPY PRE-EXTRACTION
# Avoids all df.iloc inside the hot loop.
# Called once in the main process; numpy arrays
# are passed to workers (trivial size ~1MB).
# ============================================

def extract_numpy(df):
    """Pull every column we'll ever need into contiguous numpy arrays."""
    iv_arr     = df['IV'].to_numpy(dtype=np.float32)
    target_arr = df[TARGET_COL].to_numpy(dtype=np.float32)
    static_arr = df[STATIC_FEATURES].to_numpy(dtype=np.float32)
    dates_arr  = df['CURRENT_DATE'].to_numpy()   # kept for logging

    # One array per lag_n for PCs and RMSE
    lag_pc   = {}   # lag_n -> (n_rows, N_FACTORS) float32
    lag_rmse = {}   # lag_n -> (n_rows,) float32

    for lag_n in LAG_VALUES:
        if lag_n == 0:
            pc_cols  = [f'CURRENT_PC{k}' for k in range(1, N_FACTORS + 1)]
            rmse_col = 'RMSE_CURRENT'
        else:
            pc_cols  = [f'LAG{lag_n}_PC{k}' for k in range(1, N_FACTORS + 1)]
            rmse_col = f'RMSE_LAG{lag_n}'
        lag_pc[lag_n]   = df[pc_cols].to_numpy(dtype=np.float32)
        lag_rmse[lag_n] = df[rmse_col].to_numpy(dtype=np.float32)

    return lag_pc, lag_rmse, iv_arr, static_arr, target_arr, dates_arr


# ============================================
# SEQUENCE BUILDERS — pure numpy, no pandas
# ============================================

def build_sequences_np(train_start, train_end,
                        lag_pc, lag_rmse, iv_arr, static_arr, target_arr):
    X_seq_list    = []
    X_static_list = []
    y_list        = []

    for j in range(train_start, train_end):
        if j - MAX_LAG < 0:
            continue
        if np.isnan(target_arr[j]):
            continue

        seq  = np.zeros((N_LAGS, N_SEQ_FEATURES), dtype=np.float32)
        skip = False

        for t_idx, lag_n in enumerate(LAG_VALUES):
            pc_vals  = lag_pc[lag_n][j]          # (N_FACTORS,) — no copy, view
            rmse_val = lag_rmse[lag_n][j]
            iv_val   = iv_arr[j - lag_n]

            if np.any(np.isnan(pc_vals)) or np.isnan(rmse_val) or np.isnan(iv_val):
                skip = True
                break

            seq[t_idx, :N_FACTORS] = pc_vals
            seq[t_idx,  N_FACTORS] = rmse_val
            seq[t_idx,  N_FACTORS + 1] = iv_val

        if skip:
            continue

        sv = static_arr[j]
        if np.any(np.isnan(sv)):
            continue

        X_seq_list.append(seq)
        X_static_list.append(sv)
        y_list.append(target_arr[j])

    if not y_list:
        return None, None, None

    return (np.array(X_seq_list,    dtype=np.float32),
            np.array(X_static_list, dtype=np.float32),
            np.array(y_list,        dtype=np.float32))


def build_single_np(i, lag_pc, lag_rmse, iv_arr, static_arr):
    if i - MAX_LAG < 0:
        return None, None

    seq = np.zeros((1, N_LAGS, N_SEQ_FEATURES), dtype=np.float32)

    for t_idx, lag_n in enumerate(LAG_VALUES):
        pc_vals  = lag_pc[lag_n][i]
        rmse_val = lag_rmse[lag_n][i]
        iv_val   = iv_arr[i - lag_n]

        if np.any(np.isnan(pc_vals)) or np.isnan(rmse_val) or np.isnan(iv_val):
            return None, None

        seq[0, t_idx, :N_FACTORS] = pc_vals
        seq[0, t_idx,  N_FACTORS] = rmse_val
        seq[0, t_idx,  N_FACTORS + 1] = iv_val

    sv = static_arr[i]
    if np.any(np.isnan(sv)):
        return None, None

    return seq, sv.reshape(1, -1)


# ============================================
# WORKER — runs in its own spawned process.
# TF is imported here so it never gets forked.
# ============================================

def worker_fn(worker_id, indices, numpy_data, skip_set, out_path, params):
    """
    numpy_data : tuple returned by extract_numpy()
    skip_set   : set of row indices already predicted (from checkpoint)
    out_path   : where this worker writes its partial results CSV
    params     : dict of hyper-parameters
    """
    tf.random.set_seed(27)
    np.random.seed(27)
    random.seed(27)

    # unpack
    lag_pc, lag_rmse, iv_arr, static_arr, target_arr, dates_arr = numpy_data

    TRAIN_WINDOW = params['TRAIN_WINDOW']
    FUTURE       = params['FUTURE']
    LSTM_UNITS   = params['LSTM_UNITS']
    EPOCHS       = params['EPOCHS']
    BATCH_SIZE   = params['BATCH_SIZE']
    LR           = params['LR']
    PATIENCE     = params['PATIENCE']
    N_STATIC     = params['N_STATIC']

    def build_model():
        seq_in   = Input(shape=(N_LAGS, N_SEQ_FEATURES), name='seq')
        lstm_out = LSTM(LSTM_UNITS, activation='tanh',
                        recurrent_activation='sigmoid',
                        return_sequences=False)(seq_in)
        stat_in  = Input(shape=(N_STATIC,), name='stat')
        x = Concatenate()([lstm_out, stat_in])
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        out = Dense(1, activation='sigmoid')(x)
        m = Model(inputs=[seq_in, stat_in], outputs=out)
        m.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss='binary_crossentropy')
        return m

    results   = []   # list of (date_str, true_val, pred)
    time_log  = []
    start     = time.time()
    done      = 0

    for i in indices:
        if i in skip_set:
            continue

        t0 = time.time()

        train_end   = i - FUTURE
        train_start = train_end - TRAIN_WINDOW

        if train_start < 0:
            done += 1
            continue

        X_seq_raw, X_static_raw, y_train = build_sequences_np(
            train_start, train_end,
            lag_pc, lag_rmse, iv_arr, static_arr, target_arr
        )

        if y_train is None or len(y_train) < BATCH_SIZE:
            done += 1
            continue

        # Scale
        scaler_seq    = StandardScaler()
        scaler_static = StandardScaler()
        n, t, f = X_seq_raw.shape
        X_seq_sc    = scaler_seq.fit_transform(
                          X_seq_raw.reshape(-1, f)
                      ).reshape(n, t, f).astype(np.float32)
        X_static_sc = scaler_static.fit_transform(X_static_raw).astype(np.float32)

        # Train
        model = build_model()
        es    = EarlyStopping(monitor='loss', patience=PATIENCE,
                              restore_best_weights=True, verbose=0)
        model.fit([X_seq_sc, X_static_sc], y_train,
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=[es], verbose=0)

        del X_seq_sc, X_static_sc, X_seq_raw, X_static_raw, y_train

        # Predict
        x_seq, x_stat = build_single_np(i, lag_pc, lag_rmse, iv_arr, static_arr)

        if x_seq is None:
            del model, es
            tf.keras.backend.clear_session()
            gc.collect()
            done += 1
            continue

        x_seq_sc  = scaler_seq.transform(
                        x_seq.reshape(-1, N_SEQ_FEATURES)
                    ).reshape(1, N_LAGS, N_SEQ_FEATURES)
        x_stat_sc = scaler_static.transform(x_stat)

        prob = float(model([x_seq_sc, x_stat_sc], training=False).numpy()[0][0])
        pred = 1 if prob >= 0.5 else 0
        true_val = int(target_arr[i]) if not np.isnan(target_arr[i]) else -1

        date_str = str(pd.Timestamp(dates_arr[i]).date())
        results.append({'row': i, 'date': date_str,
                        'true': true_val, 'pred': pred})

        # Full memory release
        del model, es, x_seq, x_stat, x_seq_sc, x_stat_sc
        tf.keras.backend.clear_session()
        gc.collect()

        elapsed_iter = time.time() - t0
        time_log.append(elapsed_iter)
        done += 1

        # Progress print + partial save every 100
        if done % 100 == 0:
            n_done      = len(results)
            n_correct   = sum(r['true'] == r['pred'] for r in results[-100:])
            acc_100     = n_correct / min(100, len(results))
            acc_cum     = sum(r['true'] == r['pred'] for r in results) / len(results) if results else 0
            avg_t       = np.mean(time_log[-100:])
            remaining   = len(indices) - done
            eta_s       = remaining * avg_t
            elapsed_s   = time.time() - start

            print(f"  [W{worker_id}] {done:>5}  {date_str:<12}  "
                  f"last100={acc_100:.1%}  cum={acc_cum:.1%}  "
                  f"{avg_t:.1f}s/it  "
                  f"{int(elapsed_s//3600)}h{int((elapsed_s%3600)//60):02d}m elapsed  "
                  f"ETA {int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m",
                  flush=True)

            pd.DataFrame(results).to_csv(out_path, index=False)

    # Final save
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  [W{worker_id}] Done — {len(results)} predictions written to {out_path}",
          flush=True)


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # --- Load ---
    print(f"Loading {FILE_PATH} ...")
    df = pd.read_csv(FILE_PATH)
    df['CURRENT_DATE'] = pd.to_datetime(df['CURRENT_DATE'])
    df = df.sort_values('CURRENT_DATE').reset_index(drop=True)

    if PRED_COL not in df.columns:
        df[PRED_COL] = np.nan

    print(f"Dataset shape : {df.shape}")
    print(f"Date range    : {df['CURRENT_DATE'].min().date()} to {df['CURRENT_DATE'].max().date()}")
    print(f"Target dist.  :\n{df[TARGET_COL].value_counts()}\n")

    # --- Pre-extract to numpy (once, in main process) ---
    print("Pre-extracting columns to numpy ...")
    numpy_data = extract_numpy(df)
    lag_pc, lag_rmse, iv_arr, static_arr, target_arr, dates_arr = numpy_data
    print("  Done.\n")

    # --- Determine prediction range ---
    first_pred_idx = TRAIN_WINDOW + FUTURE
    all_indices    = list(range(first_pred_idx, len(df)))

    # --- Load existing checkpoint (any worker partial files) ---
    skip_set = set()
    ckpt_main = os.path.join(CKPT_DIR, f'checkpoint_pc{TARGET_PC}.csv')

    # Collect any partial files from previous run
    partial_files = [
        os.path.join(CKPT_DIR, f'partial_pc{TARGET_PC}_w{wid}.csv')
        for wid in range(N_WORKERS)
    ]
    completed_rows = []
    for pf in [ckpt_main] + partial_files:
        if os.path.exists(pf):
            try:
                tmp = pd.read_csv(pf)
                if 'row' in tmp.columns and 'pred' in tmp.columns:
                    completed_rows.append(tmp)
                    skip_set.update(tmp['row'].tolist())
                    print(f"  Found checkpoint: {pf} ({len(tmp)} rows)")
            except Exception:
                pass

    if completed_rows:
        df_done = pd.concat(completed_rows, ignore_index=True).drop_duplicates('row')
        # Merge back into df
        row_to_pred = dict(zip(df_done['row'], df_done['pred']))
        for row_idx, pred_val in row_to_pred.items():
            if 0 <= row_idx < len(df):
                df.loc[row_idx, PRED_COL] = pred_val
        print(f"  Resuming: {len(skip_set)} predictions already done, "
              f"{len(all_indices) - len(skip_set)} remaining\n")
    else:
        print("  No checkpoint found — starting fresh\n")

    remaining_indices = [i for i in all_indices if i not in skip_set]
    total_remaining   = len(remaining_indices)

    if total_remaining == 0:
        print("All predictions already complete.")
    else:
        # --- Split into N_WORKERS chunks ---
        chunk_size = total_remaining // N_WORKERS
        chunks = []
        for wid in range(N_WORKERS):
            start_idx = wid * chunk_size
            end_idx   = (wid + 1) * chunk_size if wid < N_WORKERS - 1 else total_remaining
            chunks.append(remaining_indices[start_idx:end_idx])

        print(f"Rolling LSTM — PC{TARGET_PC}")
        print(f"  Workers          : {N_WORKERS}")
        print(f"  Total remaining  : {total_remaining}")
        print(f"  Per worker       : ~{chunk_size}")
        print(f"  Est. remaining   : {total_remaining * 7 / 3600 / N_WORKERS:.1f}–"
              f"{total_remaining * 17 / 3600 / N_WORKERS:.1f} hours")
        print(f"{'─'*80}\n")

        params = {
            'TRAIN_WINDOW': TRAIN_WINDOW,
            'FUTURE':       FUTURE,
            'LSTM_UNITS':   LSTM_UNITS,
            'EPOCHS':       EPOCHS,
            'BATCH_SIZE':   BATCH_SIZE,
            'LR':           LR,
            'PATIENCE':     PATIENCE,
            'N_STATIC':     len(STATIC_FEATURES),
        }

        # --- Spawn workers ---
        processes = []
        for wid in range(N_WORKERS):
            out_path = os.path.join(CKPT_DIR, f'partial_pc{TARGET_PC}_w{wid}.csv')
            p = mp.Process(
                target=worker_fn,
                args=(wid, chunks[wid], numpy_data, skip_set, out_path, params),
                daemon=False
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"\n{'─'*80}")
        print("All workers finished. Merging results ...")

        # --- Merge partial results back into df ---
        all_partial = []
        for wid in range(N_WORKERS):
            out_path = os.path.join(CKPT_DIR, f'partial_pc{TARGET_PC}_w{wid}.csv')
            if os.path.exists(out_path):
                try:
                    tmp = pd.read_csv(out_path)
                    all_partial.append(tmp)
                except Exception as e:
                    print(f"  Warning: could not read {out_path}: {e}")

        if all_partial:
            df_new = pd.concat(all_partial, ignore_index=True).drop_duplicates('row')
            for _, row in df_new.iterrows():
                df.loc[int(row['row']), PRED_COL] = row['pred']
            print(f"  Merged {len(df_new)} new predictions.")

    # --- Save final ---
    df.to_csv(FILE_PATH, index=False)
    print(f"✓ Saved: {FILE_PATH}")

    # Clean up partial files
    for wid in range(N_WORKERS):
        pf = os.path.join(CKPT_DIR, f'partial_pc{TARGET_PC}_w{wid}.csv')
        if os.path.exists(pf):
            os.remove(pf)
    if os.path.exists(ckpt_main):
        os.remove(ckpt_main)
    print("✓ Checkpoint files cleaned up")

    # ============================================
    # VALIDATION (no plots)
    # ============================================

    valid  = df[PRED_COL].notna()
    y_true = df.loc[valid, TARGET_COL].values.astype(int)
    y_pred = df.loc[valid, PRED_COL].values.astype(int)
    acc    = accuracy_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"VALIDATION — PC{TARGET_PC} Direction Prediction")
    print(f"{'='*60}")
    print(f"Total predictions : {len(y_pred)}")
    print(f"Overall accuracy  : {acc:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Down (0)', 'Up (1)']))

    def assign_regime(date):
        if date < pd.Timestamp('2008-09-01'):  return 'Pre-Crisis (2002-2008)'
        elif date < pd.Timestamp('2012-01-01'): return 'Crisis (2008-2012)'
        elif date < pd.Timestamp('2022-01-01'): return 'Zero Rate Era (2012-2022)'
        else:                                   return 'Hiking Cycle (2022-2026)'

    df_val = df.loc[valid].copy()
    df_val['REGIME'] = df_val['CURRENT_DATE'].apply(assign_regime)

    print(f"\nAccuracy by regime:")
    print(f"{'─'*45}")
    for regime in ['Pre-Crisis (2002-2008)', 'Crisis (2008-2012)',
                   'Zero Rate Era (2012-2022)', 'Hiking Cycle (2022-2026)']:
        mask = df_val['REGIME'] == regime
        if mask.sum() == 0:
            continue
        yt    = df_val.loc[mask, TARGET_COL].values.astype(int)
        yp    = df_val.loc[mask, PRED_COL].values.astype(int)
        r_acc = accuracy_score(yt, yp)
        print(f"  {regime:<30}: {r_acc:.1%}  (n={mask.sum()})")