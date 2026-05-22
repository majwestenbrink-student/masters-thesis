"""
Rolling LSTM — Full Lag Features + Eigenvalues (row-based)
===========================================================
Same as LSTM_FULL but adds 4 eigenvalues as static input
fed through a Dense(16) layer concatenated with LSTM output.

Architecture: LSTM(50) → LSTM(20) → Concatenate(static) → Dense(16) → Dense(4, tanh)
Sequence shape: (11 timesteps, 5 features)
Static shape  : (4,)  — EIGENVAL_1..4

Input file : /home/cdsw/LSTM_FULL_EIG_Upload.csv
Checkpoints: /home/cdsw/checkpoints-LSTM_FULL_EIG_Upload/
Output     : /home/cdsw/LSTM_DONE/LSTM_FULL_EIG_Done.csv
             (written ONLY when all predictions complete)
"""

import os, gc, sys, time, random, warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# ── Tee: mirrors stdout to log file ───────────────────────────────────────
class Tee:
    def __init__(self, path):
        self.file   = open(path, 'a', buffering=1, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout  = self
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        sys.stdout = self.stdout
        self.file.close()

# ── Configuration ──────────────────────────────────────────────────────────
CSV_NAME  = 'LSTM_FULL_EIG_Upload'
FILE_PATH = f'/home/cdsw/{CSV_NAME}.csv'
CKPT_DIR  = f'/home/cdsw/checkpoints-{CSV_NAME}'
DONE_DIR  = '/home/cdsw/LSTM_DONE'
DONE_PATH = f'{DONE_DIR}/LSTM_FULL_EIG_Done.csv'
LOG_PATH  = f'{CKPT_DIR}/lstm_full_eig_log.txt'

PCS          = [1, 2, 3, 4]
N_FACTORS    = 4
HORIZON      = 21
TRAIN_WINDOW = 3000

# Timestep order: oldest → newest
TARGET_LAGS  = [211, 190, 169, 148, 127, 106, 85, 64, 43, 22]
ALL_STEPS    = TARGET_LAGS + ['current']   # 11 timesteps
N_TIMESTEPS  = len(ALL_STEPS)             # 11
N_SEQ_FEAT   = 5                          # PC1, PC2, PC3, PC4, RMSE
N_STATIC     = 4                          # EIGENVAL_1..4
DENSE_UNITS  = 16

LSTM_UNITS  = 50
EPOCHS      = 500
PATIENCE    = 40
BATCH_SIZE  = 32
LR          = 0.001
N_WORKERS   = 16
SEED        = 27

TARGET_COLS = [f'FUTURE_PC{pc}_DIRECTION_DIFF' for pc in PCS]
STATIC_COLS = ['EIGENVAL_1', 'EIGENVAL_2', 'EIGENVAL_3', 'EIGENVAL_4']
PRED_COLS   = [f'LSTM_PC{pc}' for pc in PCS]

# Pre-computed column names per timestep (ordered oldest→newest)
# seq_col_map[step] = list of 5 column names [PC1, PC2, PC3, PC4, RMSE]
def build_col_map():
    col_map = {}
    for lag in TARGET_LAGS:
        col_map[lag] = (
            [f'LAG{lag}_PC{pc}' for pc in PCS] + [f'RMSE_LAG{lag}']
        )
    col_map['current'] = (
        [f'CURRENT_PC{pc}' for pc in PCS] + ['RMSE_CURRENT']
    )
    return col_map

SEQ_COL_MAP = build_col_map()

# All required input columns
ALL_SEQ_COLS = []
for step in ALL_STEPS:
    ALL_SEQ_COLS.extend(SEQ_COL_MAP[step])


# ── Worker ─────────────────────────────────────────────────────────────────
def worker_fn(worker_id, indices, numpy_data, skip_set, out_path, params):
    import os, gc, random, time, traceback
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler

    tf.random.set_seed(params['SEED'])
    np.random.seed(params['SEED'])
    random.seed(params['SEED'])

    # numpy_data['seq_arr'] shape: (N, N_TIMESTEPS, N_SEQ_FEAT)
    seq_arr    = numpy_data['seq_arr']
    static_arr = numpy_data['static_arr']
    target_arr = numpy_data['target_arr']
    dates_arr  = numpy_data['dates_arr']
    scale_vec  = numpy_data['scale_vec']

    TRAIN_WINDOW = params['TRAIN_WINDOW']
    HORIZON      = params['HORIZON']
    N_TIMESTEPS  = params['N_TIMESTEPS']
    N_SEQ_FEAT   = params['N_SEQ_FEAT']
    N_STATIC     = params['N_STATIC']
    N_FACTORS    = params['N_FACTORS']
    LSTM_UNITS   = params['LSTM_UNITS']
    DENSE_UNITS  = params['DENSE_UNITS']
    EPOCHS       = params['EPOCHS']
    PATIENCE     = params['PATIENCE']
    BATCH_SIZE   = params['BATCH_SIZE']
    LR           = params['LR']
    PCS          = [1, 2, 3, 4]

    log_path   = out_path.replace('.csv', '_log.txt')
    log_handle = open(log_path, 'a', encoding='utf-8', buffering=1)

    def wprint(msg):
        full = f'[W{worker_id}] {msg}'
        print(full, flush=True)
        log_handle.write(full + '\n')
        log_handle.flush()

    wprint(f'START  n={len(indices)}  first={indices[0] if indices else "none"}')

    def build_sequences(start, end):
        """Read pre-computed rows directly — no shifting needed."""
        X_seq_list, X_stat_list, y_list = [], [], []
        for j in range(start, end):
            seq = seq_arr[j]
            if np.any(np.isnan(seq)):
                continue
            sv = static_arr[j]
            if np.any(np.isnan(sv)):
                continue
            yv = target_arr[j]
            if np.any(np.isnan(yv)):
                continue
            X_seq_list.append(seq)
            X_stat_list.append(sv)
            y_list.append(yv)
        if not y_list:
            return None, None, None
        return (np.array(X_seq_list,  dtype=np.float32),
                np.array(X_stat_list, dtype=np.float32),
                np.array(y_list,      dtype=np.float32))

    def build_single(i):
        seq = seq_arr[i]
        if np.any(np.isnan(seq)):
            return None, None
        sv = static_arr[i]
        if np.any(np.isnan(sv)):
            return None, None
        return seq[np.newaxis], sv.reshape(1, -1)

    def build_model():
        seq_in   = Input(shape=(N_TIMESTEPS, N_SEQ_FEAT), name='seq')
        x        = LSTM(LSTM_UNITS, activation='tanh',
                        recurrent_activation='sigmoid',
                        return_sequences=True)(seq_in)
        x        = LSTM(20, activation='tanh',
                        recurrent_activation='sigmoid',
                        return_sequences=False)(x)
        stat_in  = Input(shape=(N_STATIC,), name='static')
        x        = Concatenate()([x, stat_in])
        x        = Dense(DENSE_UNITS, activation='relu')(x)
        x        = Dropout(0.2)(x)
        out      = Dense(N_FACTORS, activation='tanh', name='output')(x)
        model    = Model(inputs=[seq_in, stat_in], outputs=out)
        model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='mae')
        return model

    results  = []
    time_log = []
    start    = time.time()
    done     = 0

    try:
        for i in indices:
            if i in skip_set:
                done += 1; continue

            t0          = time.time()
            train_end   = i - HORIZON
            train_start = train_end - TRAIN_WINDOW
            if train_start < 0:
                done += 1; continue

            if done % 10 == 0:
                elapsed_m = int((time.time() - start) // 60)
                wprint(f'row={i}  done={done}/{len(indices)}  elapsed={elapsed_m}m')

            X_seq_raw, X_stat_raw, y_raw = build_sequences(train_start, train_end)
            if y_raw is None or len(y_raw) < BATCH_SIZE:
                done += 1; continue

            y_norm      = np.clip(y_raw / scale_vec, -1.0, 1.0).astype(np.float32)
            scaler      = StandardScaler()
            scaler_stat = StandardScaler()
            n, t, f     = X_seq_raw.shape
            X_seq_sc    = scaler.fit_transform(
                              X_seq_raw.reshape(-1, f)
                          ).reshape(n, t, f).astype(np.float32)
            X_stat_sc   = scaler_stat.fit_transform(X_stat_raw).astype(np.float32)

            model = build_model()
            es    = EarlyStopping(monitor='loss', patience=PATIENCE,
                                  restore_best_weights=True, verbose=0)
            rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                     patience=10, min_lr=1e-5, verbose=0)
            hist  = model.fit(
                [X_seq_sc, X_stat_sc], y_norm,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=[es, rlrop], verbose=0
            )

            epochs_ran    = len(hist.history['loss'])
            best_tr_loss  = min(hist.history['loss'])
            stopped_early = int(epochs_ran < EPOCHS)

            del X_seq_sc, X_stat_sc, X_seq_raw, X_stat_raw, y_raw, y_norm

            x_seq, x_stat = build_single(i)
            if x_seq is None:
                del model, es; tf.keras.backend.clear_session(); gc.collect()
                done += 1; continue

            x_seq_sc  = scaler.transform(
                            x_seq.reshape(-1, N_SEQ_FEAT)
                        ).reshape(1, N_TIMESTEPS, N_SEQ_FEAT)
            x_stat_sc = scaler_stat.transform(x_stat)

            pred_norm = model([x_seq_sc, x_stat_sc], training=False).numpy()[0]
            preds     = pred_norm * scale_vec
            true_vals = target_arr[i]
            date_str  = str(pd.Timestamp(dates_arr[i]).date())

            row = {'row': i, 'date': date_str,
                   'epochs_ran': epochs_ran,
                   'stopped_early': stopped_early,
                   'best_tr_loss': round(best_tr_loss, 6)}
            for k, pc in enumerate(PCS):
                row[f'pred_pc{pc}']   = float(preds[k])
                row[f'actual_pc{pc}'] = float(true_vals[k])
            results.append(row)

            del model, es, x_seq, x_stat, x_seq_sc, x_stat_sc, pred_norm
            tf.keras.backend.clear_session(); gc.collect()

            time_log.append(time.time() - t0)
            done += 1

            if len(results) == 1 or done % 50 == 0:
                pd.DataFrame(results).to_csv(out_path, index=False)

            if done % 50 == 0:
                avg_t     = np.mean(time_log[-50:])
                remaining = len(indices) - done
                eta_s     = remaining * avg_t
                elapsed_s = time.time() - start
                recent    = results[-50:]
                dir_strs  = ''
                for pc in PCS:
                    yt = np.array([r[f'actual_pc{pc}'] for r in recent])
                    yp = np.array([r[f'pred_pc{pc}']   for r in recent])
                    da = np.mean(np.sign(yt) == np.sign(yp))
                    dir_strs += f'PC{pc}={da:.1%} '
                wprint(
                    f'{done:>4}/{len(indices)}  {date_str}  '
                    f'dir: {dir_strs}'
                    f'ep={np.mean([r["epochs_ran"] for r in recent]):.0f} '
                    f'ES={sum(r["stopped_early"] for r in recent)}/50 '
                    f'tr_loss={np.mean([r["best_tr_loss"] for r in recent]):.5f}  '
                    f'{avg_t:.1f}s/it  '
                    f'elapsed={int(elapsed_s//3600)}h{int((elapsed_s%3600)//60):02d}m  '
                    f'ETA={int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m'
                )

        pd.DataFrame(results).to_csv(out_path, index=False)
        wprint(f'DONE -- {len(results)} predictions written.')

    except Exception:
        err = traceback.format_exc()
        wprint(f'CRASHED:\n{err}')
        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)
            wprint(f'Saved {len(results)} partial results before crash.')
    finally:
        log_handle.close()


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)
    tee = Tee(LOG_PATH)

    print('='*70)
    print('Rolling LSTM — Full Lag Features + Eigenvalues (row-based)')
    print(f'  File       : {FILE_PATH}')
    print(f'  Checkpoints: {CKPT_DIR}')
    print(f'  Output     : {DONE_PATH}')
    print(f'  Train win  : {TRAIN_WINDOW}  |  Horizon : {HORIZON}')
    print(f'  Timesteps  : {ALL_STEPS}')
    print(f'  Seq shape  : ({N_TIMESTEPS}, {N_SEQ_FEAT})')
    print(f'  Static dim : {N_STATIC}  (eigenvalues)')
    print(f'  LSTM units : {LSTM_UNITS} → 20 → Dense({DENSE_UNITS})  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}')
    print(f'  Workers    : {N_WORKERS}')
    print('='*70, flush=True)

    # Load
    print('\nLoading data...')
    df = pd.read_csv(FILE_PATH)
    df['REFERENCE_DATE'] = pd.to_datetime(df['REFERENCE_DATE'])
    df = df.sort_values('REFERENCE_DATE').reset_index(drop=True)
    N  = len(df)
    print(f'  Shape: {df.shape}')
    print(f'  Dates: {df["REFERENCE_DATE"].min().date()} to {df["REFERENCE_DATE"].max().date()}')

    # Verify all required columns exist
    missing = [c for c in ALL_SEQ_COLS + TARGET_COLS + STATIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    print(f'  All {len(ALL_SEQ_COLS)} sequence columns and {len(STATIC_COLS)} static columns confirmed.')

    for col in PRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Scale
    scale_vec = np.array([
        float(max(df[TARGET_COLS[pc-1]].max(), -df[TARGET_COLS[pc-1]].min()))
        for pc in PCS
    ], dtype=np.float32)
    for pc in PCS:
        print(f'  PC{pc} scale = {scale_vec[pc-1]:.6f}')

    # Pre-extract 3D array: (N, N_TIMESTEPS, N_SEQ_FEAT)
    print('\nPre-extracting sequence array...')
    seq_arr = np.stack([
        df[SEQ_COL_MAP[step]].to_numpy(dtype=np.float32)
        for step in ALL_STEPS
    ], axis=1)   # (N, 11, 5)
    print(f'  seq_arr shape: {seq_arr.shape}')

    target_arr = df[TARGET_COLS].to_numpy(dtype=np.float32)
    static_arr = df[STATIC_COLS].to_numpy(dtype=np.float32)
    dates_arr  = df['REFERENCE_DATE'].to_numpy()

    # FIRST_PRED: need TRAIN_WINDOW rows before t, each row needs lag211 valid
    # lag211 columns are NaN for rows 0..210, so first usable training row = 211
    # → FIRST_PRED = TRAIN_WINDOW + HORIZON + 211 = 3232 (consistent with VAR)
    MAX_LAG    = max(TARGET_LAGS)
    FIRST_PRED = TRAIN_WINDOW + HORIZON + MAX_LAG
    print(f'  First prediction at row {FIRST_PRED}  ({pd.Timestamp(dates_arr[FIRST_PRED]).date()})')
    print(f'  Total predictions: {N - FIRST_PRED}')

    # Checkpoints
    ckpt_main     = os.path.join(CKPT_DIR, 'lstm_full_eig_checkpoint.csv')
    partial_paths = [os.path.join(CKPT_DIR, f'lstm_full_eig_partial_w{w}.csv')
                     for w in range(N_WORKERS)]

    skip_set  = set()
    completed = []
    for path in [ckpt_main] + partial_paths:
        if os.path.exists(path):
            try:
                tmp = pd.read_csv(path)
                if 'row' in tmp.columns:
                    completed.append(tmp)
                    skip_set.update(tmp['row'].tolist())
                    print(f'  Checkpoint: {path} ({len(tmp)} rows)')
            except Exception:
                pass

    if completed:
        df_done     = pd.concat(completed, ignore_index=True).drop_duplicates('row')
        row_to_pred = {int(r['row']): r for _, r in df_done.iterrows()}
        for row_idx, r in row_to_pred.items():
            if 0 <= row_idx < N:
                for pc in PCS:
                    df.loc[row_idx, f'LSTM_PC{pc}'] = r[f'pred_pc{pc}']
        print(f'  Resuming: {len(skip_set)} done, {N - FIRST_PRED - len(skip_set)} remaining.')
    else:
        print('  No checkpoint — starting fresh.')

    all_indices       = list(range(FIRST_PRED, N))
    remaining_indices = [i for i in all_indices if i not in skip_set]
    total_remaining   = len(remaining_indices)

    if total_remaining == 0:
        print('All predictions already complete.')
    else:
        numpy_data = {
            'seq_arr':    seq_arr,
            'static_arr': static_arr,
            'target_arr': target_arr,
            'dates_arr':  dates_arr,
            'scale_vec':  scale_vec,
        }
        params = {
            'TRAIN_WINDOW': TRAIN_WINDOW, 'HORIZON':     HORIZON,
            'N_TIMESTEPS':  N_TIMESTEPS,  'N_SEQ_FEAT':  N_SEQ_FEAT,
            'N_STATIC':     N_STATIC,     'N_FACTORS':   N_FACTORS,
            'LSTM_UNITS':   LSTM_UNITS,   'DENSE_UNITS': DENSE_UNITS,
            'EPOCHS':       EPOCHS,       'PATIENCE':    PATIENCE,
            'BATCH_SIZE':   BATCH_SIZE,   'LR':          LR,
            'SEED':         SEED,
        }

        chunk_size = total_remaining // N_WORKERS
        chunks = []
        for w in range(N_WORKERS):
            s = w * chunk_size
            e = (w + 1) * chunk_size if w < N_WORKERS - 1 else total_remaining
            chunks.append(remaining_indices[s:e])

        print(f'\nRolling LSTM — {N_WORKERS} workers')
        print(f'  Total remaining : {total_remaining}')
        print(f'  Per worker      : ~{chunk_size}')
        est_low  = total_remaining * 5  / 3600 / N_WORKERS
        est_high = total_remaining * 15 / 3600 / N_WORKERS
        print(f'  Est. time       : {est_low:.1f}–{est_high:.1f} hours')
        print(f'{"─"*70}\n', flush=True)

        processes = []
        for w in range(N_WORKERS):
            p = mp.Process(
                target=worker_fn,
                args=(w, chunks[w], numpy_data, skip_set, partial_paths[w], params),
                daemon=False
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f'\n{"─"*70}')
        print('All workers finished. Merging...')

        all_partial = []
        for path in partial_paths:
            if os.path.exists(path):
                try:
                    all_partial.append(pd.read_csv(path))
                except Exception as e:
                    print(f'  Warning: could not read {path}: {e}')

        if all_partial:
            df_new = pd.concat(all_partial, ignore_index=True).drop_duplicates('row')
            for _, r in df_new.iterrows():
                idx = int(r['row'])
                for pc in PCS:
                    df.loc[idx, f'LSTM_PC{pc}'] = r[f'pred_pc{pc}']
            df_new.to_csv(ckpt_main, index=False)
            print(f'  Merged {len(df_new)} predictions.')

        for path in partial_paths:
            if os.path.exists(path):
                os.remove(path)

    # Only write to LSTM_DONE when fully complete
    n_filled   = df[PRED_COLS].notna().all(axis=1).sum()
    n_expected = N - FIRST_PRED
    if n_filled >= n_expected:
        df.to_csv(DONE_PATH, index=False)
        print(f'\nFully complete — saved to {DONE_PATH}')
    else:
        print(f'\nNot yet complete ({n_filled}/{n_expected}) — NOT uploading to LSTM_DONE.')
        print('Re-run to resume from checkpoints.')

    print(f'\n{"="*70}')
    print('DONE')
    print(f'{"="*70}', flush=True)
    tee.close()