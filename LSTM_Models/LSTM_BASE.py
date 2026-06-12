"""
Rolling LSTM — Base Model (15 tenor 21-day returns)
=====================================================
Naive base case: uses only 21-day returns of all 15 tenors, lagged,
to predict the 21-day forward return of all 15 tenors.

Architecture: LSTM(50) → LSTM(20) → Dense(15, tanh)

Input file : /home/cdsw/LSTM_BASE_Upload.csv
Checkpoints: /home/cdsw/checkpoints-LSTM_BASE/
Output     : /home/cdsw/LSTM_DONE/LSTM_BASE_Done.csv
             (written ONLY when all predictions complete)

Sequence shape: (11 timesteps, 15 features)
  Timesteps: lag211, lag190, ..., lag22, current(lag0)
  Features:  RET_ZERO_{mat} shifted by lag

No lookahead: RET_ZERO_{mat}[t-L] = yield[t-L] - yield[t-L-21].
  For L=0  : ends at t,    realised. ✓
  For L>=22 : ends at t-22, realised. ✓
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
from tensorflow.keras.layers import LSTM, Dense, Input
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
CSV_NAME  = 'LSTM_BASE_Upload'
FILE_PATH = f'/home/cdsw/{CSV_NAME}.csv'
CKPT_DIR  = f'/home/cdsw/checkpoints-LSTM_BASE'
DONE_DIR  = '/home/cdsw/LSTM_DONE'
DONE_PATH = f'{DONE_DIR}/LSTM_BASE_Done.csv'
LOG_PATH  = f'{CKPT_DIR}/lstm_base_log.txt'

MATURITY_NAMES = ['1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y',
                  '10Y','12Y','15Y','20Y','25Y','30Y']
N_TENORS     = len(MATURITY_NAMES)          # 15
HORIZON      = 21
TRAIN_WINDOW = 3000
TARGET_LAGS  = [211, 190, 169, 148, 127, 106, 85, 64, 43, 22, 0]
N_TIMESTEPS  = len(TARGET_LAGS)             # 11
N_SEQ_FEAT   = N_TENORS                     # 15

LSTM_UNITS  = 50
EPOCHS      = 500
PATIENCE    = 40
BATCH_SIZE  = 32
LR          = 0.001
N_WORKERS   = 16
SEED        = 27

FEATURE_COLS = [f'RET_ZERO_{m}'   for m in MATURITY_NAMES]
TARGET_COLS  = [f'FUTURE_RET_{m}' for m in MATURITY_NAMES]
PRED_COLS    = [f'ZERO_{m}_PRED'  for m in MATURITY_NAMES]


# ── Worker ─────────────────────────────────────────────────────────────────
def worker_fn(worker_id, indices, numpy_data, skip_set, out_path, params):
    import os, gc, random, time, traceback
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler

    tf.random.set_seed(params['SEED'])
    np.random.seed(params['SEED'])
    random.seed(params['SEED'])

    seq_arrays  = numpy_data['seq_arrays']
    target_arr  = numpy_data['target_arr']
    dates_arr   = numpy_data['dates_arr']
    scale_vec   = numpy_data['scale_vec']
    all_lags    = numpy_data['all_lags']

    TRAIN_WINDOW = params['TRAIN_WINDOW']
    HORIZON      = params['HORIZON']
    N_TIMESTEPS  = params['N_TIMESTEPS']
    N_SEQ_FEAT   = params['N_SEQ_FEAT']
    N_TENORS     = params['N_TENORS']
    LSTM_UNITS   = params['LSTM_UNITS']
    EPOCHS       = params['EPOCHS']
    PATIENCE     = params['PATIENCE']
    BATCH_SIZE   = params['BATCH_SIZE']
    LR           = params['LR']

    log_path   = out_path.replace('.csv', '_log.txt')
    log_handle = open(log_path, 'a', encoding='utf-8', buffering=1)

    def wprint(msg):
        full = f'[W{worker_id}] {msg}'
        print(full, flush=True)
        log_handle.write(full + '\n')
        log_handle.flush()

    wprint(f'START  n={len(indices)}  first={indices[0] if indices else "none"}')

    def build_sequences(start, end):
        X_seq_list, y_list = [], []
        for j in range(start, end):
            seq  = np.empty((N_TIMESTEPS, N_SEQ_FEAT), dtype=np.float32)
            skip = False
            for t_i, lag in enumerate(all_lags):
                vals = seq_arrays[lag][j]
                if np.any(np.isnan(vals)):
                    skip = True; break
                seq[t_i] = vals
            if skip:
                continue
            yv = target_arr[j]
            if np.any(np.isnan(yv)):
                continue
            X_seq_list.append(seq)
            y_list.append(yv)
        if not y_list:
            return None, None
        return (np.array(X_seq_list, dtype=np.float32),
                np.array(y_list,     dtype=np.float32))

    def build_single(i):
        seq = np.empty((1, N_TIMESTEPS, N_SEQ_FEAT), dtype=np.float32)
        for t_i, lag in enumerate(all_lags):
            vals = seq_arrays[lag][i]
            if np.any(np.isnan(vals)):
                return None
            seq[0, t_i] = vals
        return seq

    def build_model():
        seq_in = Input(shape=(N_TIMESTEPS, N_SEQ_FEAT), name='seq')
        x      = LSTM(LSTM_UNITS, activation='tanh',
                      recurrent_activation='sigmoid',
                      return_sequences=True)(seq_in)
        x      = LSTM(20, activation='tanh',
                      recurrent_activation='sigmoid',
                      return_sequences=False)(x)
        out    = Dense(N_TENORS, activation='tanh', name='output')(x)
        model  = Model(inputs=seq_in, outputs=out)
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

            X_seq_raw, y_raw = build_sequences(train_start, train_end)
            if y_raw is None or len(y_raw) < BATCH_SIZE:
                done += 1; continue

            # Scale targets into [-1, 1] via per-tenor scale_vec (same as VAR)
            y_norm   = np.clip(y_raw / scale_vec, -1.0, 1.0).astype(np.float32)
            scaler   = StandardScaler()
            n, t, f  = X_seq_raw.shape
            X_seq_sc = scaler.fit_transform(
                           X_seq_raw.reshape(-1, f)
                       ).reshape(n, t, f).astype(np.float32)

            model = build_model()
            es    = EarlyStopping(monitor='loss', patience=PATIENCE,
                                  restore_best_weights=True, verbose=0)
            rlrop = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                     patience=10, min_lr=1e-5, verbose=0)
            hist  = model.fit(
                X_seq_sc, y_norm,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=[es, rlrop], verbose=0
            )

            epochs_ran    = len(hist.history['loss'])
            best_tr_loss  = min(hist.history['loss'])
            stopped_early = int(epochs_ran < EPOCHS)

            del X_seq_sc, X_seq_raw, y_raw, y_norm

            x_seq = build_single(i)
            if x_seq is None:
                del model, es; tf.keras.backend.clear_session(); gc.collect()
                done += 1; continue

            x_seq_sc  = scaler.transform(
                            x_seq.reshape(-1, N_SEQ_FEAT)
                        ).reshape(1, N_TIMESTEPS, N_SEQ_FEAT)

            pred_norm = model(x_seq_sc, training=False).numpy()[0]
            preds     = pred_norm * scale_vec          # back to return space
            true_vals = target_arr[i]
            date_str  = str(pd.Timestamp(dates_arr[i]).date())

            row = {'row': i, 'date': date_str,
                   'epochs_ran': epochs_ran,
                   'stopped_early': stopped_early,
                   'best_tr_loss': round(best_tr_loss, 6)}
            for k, mat in enumerate(MATURITY_NAMES):
                row[f'pred_{mat}']   = float(preds[k])
                row[f'actual_{mat}'] = float(true_vals[k])
            results.append(row)

            del model, es, x_seq, x_seq_sc, pred_norm
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
                if recent:
                    for mat in MATURITY_NAMES[:4]:   # log first 4 tenors
                        yt = np.array([r[f'actual_{mat}'] for r in recent])
                        yp = np.array([r[f'pred_{mat}']   for r in recent])
                        da = np.mean(np.sign(yt) == np.sign(yp))
                        dir_strs += f'{mat}={da:.1%} '
                wprint(
                    f'{done:>4}/{len(indices)}  {date_str}  '
                    f'dir: {dir_strs}'
                    f'ep={np.mean([r["epochs_ran"] for r in results[-50:]]):.0f} '
                    f'ES={sum(r["stopped_early"] for r in results[-50:])}/50 '
                    f'tr_loss={np.mean([r["best_tr_loss"] for r in results[-50:]]):.5f}  '
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
    print('Rolling LSTM — Base Model (15 tenor returns)')
    print(f'  File       : {FILE_PATH}')
    print(f'  Checkpoints: {CKPT_DIR}')
    print(f'  Output     : {DONE_PATH}')
    print(f'  Train win  : {TRAIN_WINDOW}  |  Horizon : {HORIZON}')
    print(f'  Lags       : {TARGET_LAGS}')
    print(f'  Seq shape  : ({N_TIMESTEPS}, {N_SEQ_FEAT})')
    print(f'  LSTM units : {LSTM_UNITS} → 20  |  Epochs : {EPOCHS}  |  Patience : {PATIENCE}')
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

    missing = [c for c in FEATURE_COLS + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    for col in PRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Scale: per-tenor max absolute return over full series (same logic as VAR)
    scale_vec = np.array([
        float(max(df[TARGET_COLS[k]].max(), -df[TARGET_COLS[k]].min()))
        for k in range(N_TENORS)
    ], dtype=np.float32)
    for k, mat in enumerate(MATURITY_NAMES):
        print(f'  {mat} scale = {scale_vec[k]:.6f}')

    # Pre-extract arrays
    print('\nPre-extracting arrays...')
    all_lags = TARGET_LAGS   # [211, 190, ..., 22, 0]

    base = df[FEATURE_COLS].to_numpy(dtype=np.float32)   # (N, 15)

    seq_arrays = {}
    for lag in TARGET_LAGS:
        if lag == 0:
            seq_arrays[0] = base.copy()
        else:
            shifted        = np.full((N, N_TENORS), np.nan, dtype=np.float32)
            shifted[lag:]  = base[:N - lag]
            seq_arrays[lag] = shifted

    target_arr = df[TARGET_COLS].to_numpy(dtype=np.float32)
    dates_arr  = df['REFERENCE_DATE'].to_numpy()

    MAX_LAG    = max(l for l in TARGET_LAGS if l > 0)    # 211
    FIRST_PRED = TRAIN_WINDOW + HORIZON + MAX_LAG         # 3232
    print(f'  First prediction at row {FIRST_PRED}  ({pd.Timestamp(dates_arr[FIRST_PRED]).date()})')
    print(f'  Total predictions: {N - FIRST_PRED}')

    # Checkpoints
    ckpt_main     = os.path.join(CKPT_DIR, 'lstm_base_checkpoint.csv')
    partial_paths = [os.path.join(CKPT_DIR, f'lstm_base_partial_w{w}.csv')
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
                for mat in MATURITY_NAMES:
                    df.loc[row_idx, f'ZERO_{mat}_PRED'] = r[f'pred_{mat}']
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
            'seq_arrays': seq_arrays,
            'target_arr': target_arr,
            'dates_arr':  dates_arr,
            'scale_vec':  scale_vec,
            'all_lags':   all_lags,
        }
        params = {
            'TRAIN_WINDOW': TRAIN_WINDOW, 'HORIZON':    HORIZON,
            'N_TIMESTEPS':  N_TIMESTEPS,  'N_SEQ_FEAT': N_SEQ_FEAT,
            'N_TENORS':     N_TENORS,     'LSTM_UNITS': LSTM_UNITS,
            'EPOCHS':       EPOCHS,       'PATIENCE':   PATIENCE,
            'BATCH_SIZE':   BATCH_SIZE,   'LR':         LR,
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
                for mat in MATURITY_NAMES:
                    df.loc[idx, f'ZERO_{mat}_PRED'] = r[f'pred_{mat}']
            df_new.to_csv(ckpt_main, index=False)
            print(f'  Merged {len(df_new)} predictions.')

        for path in partial_paths:
            if os.path.exists(path):
                os.remove(path)

    # Only write Done file when fully complete
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