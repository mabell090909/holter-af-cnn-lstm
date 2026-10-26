import os
import glob
import json
import math
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AFDB_DIR = os.path.join(BASE_DIR, 'raw_data', 'afdb')
MODEL_PATH = os.path.join(BASE_DIR, 'cnn_lstm_model.keras')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, 'cnn_lstm_model.h5')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METADATA_PATH = os.path.join(RESULTS_DIR, 'final_model_metadata.json')

TARGET_FS = 128
WINDOW_SEC = 10
WINDOW_SIZE = TARGET_FS * WINDOW_SEC

def apply_bandpass_filter(data, fs=250, lowcut=0.5, highcut=40.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def process_afdb_record(rec_path):
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')
    fs = record.fs # 250 Hz di afdb
    sig = record.p_signal[:, 0]
    
    if np.any(np.isnan(sig)) or np.any(np.isinf(sig)):
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        
    # 1. Filter bandpass sebelum resampling
    sig_filtered = apply_bandpass_filter(sig, fs=fs)
    
    # 2. Resampling polyphase (250 Hz -> 128 Hz: gcd(128, 250)=2 -> up=64, down=125)
    g = math.gcd(TARGET_FS, int(round(fs)))
    up = TARGET_FS // g
    down = int(round(fs)) // g
    sig_resampled = resample_poly(sig_filtered, up, down)
    ann_samples = np.round(ann.sample * up / down).astype(int)
    
    # 3. Anotasi ritme dengan logika non-AF yang benar
    labels = np.zeros(len(sig_resampled), dtype=np.int32)
    current_label = 0
    aux_notes = getattr(ann, 'aux_note', None)
    
    for i, sample in enumerate(ann_samples):
        note = aux_notes[i] if aux_notes is not None and i < len(aux_notes) else ''
        clean = note.replace('\x00', '').strip().upper() if isinstance(note, str) else ''
        if clean.startswith('('):
            current_label = 1 if clean.startswith(('(AFIB', '(AFL')) else 0
            
        next_sample = ann_samples[i + 1] if i + 1 < len(ann_samples) else len(sig_resampled)
        start_idx = max(0, min(sample, len(labels)))
        end_idx = max(0, min(next_sample, len(labels)))
        if start_idx < end_idx:
            labels[start_idx:end_idx] = current_label
            
    X_segs, y_segs = [], []
    for i in range(0, len(sig_resampled) - WINDOW_SIZE + 1, WINDOW_SIZE):
        w_sig = sig_resampled[i : i + WINDOW_SIZE]
        w_lbl = labels[i : i + WINDOW_SIZE]
        std_val = np.std(w_sig)
        norm_sig = (w_sig - np.mean(w_sig)) / std_val if std_val >= 1e-6 else (w_sig - np.mean(w_sig))
        is_afib = 1 if np.mean(w_lbl) >= 0.5 else 0
        X_segs.append(norm_sig.astype(np.float32))
        y_segs.append(is_afib)
        
    return X_segs, y_segs

def compute_bootstrap_ci(patient_df, n_bootstraps=1000, random_seed=0):
    """
    Bootstrap 95% Confidence Interval pada tingkat pasien (Patient-level sampling).
    Menggunakan np.nanmean sehingga metrik pasien yang undefined diabaikan secara adil.
    """
    rng = np.random.default_rng(random_seed)
    n_patients = len(patient_df)
    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    
    ci_results = {}
    for m in metrics:
        if m not in patient_df.columns:
            continue
        values = patient_df[m].to_numpy()
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            ci_results[m] = {'mean': 0.0, 'std': 0.0, 'ci_lower_95': 0.0, 'ci_upper_95': 0.0, 'n_evaluated_patients': 0}
            continue
            
        boot_means = []
        for _ in range(n_bootstraps):
            sample_idx = rng.choice(len(patient_df), size=n_patients, replace=True)
            sampled_vals = patient_df.iloc[sample_idx][m].to_numpy()
            boot_means.append(np.nanmean(sampled_vals))
            
        boot_means = np.array(boot_means)
        boot_means = boot_means[~np.isnan(boot_means)]
        
        ci_results[m] = {
            'mean': float(round(np.nanmean(values), 4)),
            'std': float(round(np.nanstd(values), 4)),
            'ci_lower_95': float(round(np.percentile(boot_means, 2.5), 4)),
            'ci_upper_95': float(round(np.percentile(boot_means, 97.5), 4)),
            'n_evaluated_patients': int(len(valid_values))
        }
    return ci_results

def find_record_dir(root):
    candidate = root
    max_dats = 0
    for dirpath, _, files in os.walk(root):
        dats = sum(1 for f in files if f.endswith('.dat'))
        if dats > max_dats:
            max_dats = dats
            candidate = dirpath
    if max_dats > 0:
        return candidate
    for dirpath, _, files in os.walk(root):
        if any(f.endswith('.hea') for f in files):
            return dirpath
    return root

def run_external_validation():
    print("=======================================================================")
    print(" Validasi Eksternal Independen (Zero-Shot / Tanpa Tuning)")
    print(" Benchmark Dataset: PhysioNet MIT-BIH Atrial Fibrillation Database (afdb)")
    print(f" Model Terlatih   : {MODEL_PATH}")
    print("=======================================================================")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model {MODEL_PATH} belum dilatih. Jalankan 2_train_model.py terlebih dahulu.")
        return
        
    # Ambil threshold final terkalibrasi dari CV
    calibrated_th = 0.50
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            meta = json.load(f)
            calibrated_th = meta.get('median_optimal_threshold', 0.50)
            print(f"[INFO] Optimal Threshold Calibrated dari CV: T* = {calibrated_th:.4f}")
            
    os.makedirs(AFDB_DIR, exist_ok=True)
    target_afdb_dir = find_record_dir(AFDB_DIR)
    hea_files = glob.glob(os.path.join(target_afdb_dir, "*.hea"))
    
    # Jika afdb belum ada di disk, unduh seluruh rekaman via wfdb
    if len(hea_files) == 0:
        print(f"[INFO] Direktori {AFDB_DIR} belum memiliki rekaman lokal.")
        print("[INFO] Mengunduh seluruh rekaman PhysioNet MIT-BIH AF Database (afdb)...")
        try:
            wfdb.dl_database('afdb', AFDB_DIR)
            target_afdb_dir = find_record_dir(AFDB_DIR)
            hea_files = glob.glob(os.path.join(target_afdb_dir, "*.hea"))
        except Exception as e:
            print(f"[PERINGATAN] wfdb.dl_database gagal ({e}), mencoba mengunduh record list...")
            try:
                rec_list = wfdb.get_record_list('afdb')
                for r in rec_list:
                    wfdb.dl_files('afdb', AFDB_DIR, [f"{r}.hea", f"{r}.dat", f"{r}.atr"])
                target_afdb_dir = find_record_dir(AFDB_DIR)
                hea_files = glob.glob(os.path.join(target_afdb_dir, "*.hea"))
            except Exception as e2:
                print(f"[ERROR] Gagal mengunduh afdb: {e2}")
                return
                
    valid_records = []
    for hea in hea_files:
        rec_id = os.path.splitext(os.path.basename(hea))[0]
        rec_path = os.path.join(target_afdb_dir, rec_id)
        if os.path.exists(f"{rec_path}.dat") and os.path.exists(f"{rec_path}.atr"):
            if os.path.getsize(f"{rec_path}.dat") > 0 and os.path.getsize(f"{rec_path}.atr") > 0:
                valid_records.append(rec_id)
                
    valid_records = sorted(valid_records)
    print(f"\nTotal Rekaman Valid di AFDB: {len(valid_records)} rekaman {valid_records}")
    
    if len(valid_records) == 0:
        print("[ERROR] Tidak ada rekaman AFDB yang memiliki sinyal lengkap.")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    all_X, all_y, all_subs = [], [], []
    for rec_id in valid_records:
        rec_path = os.path.join(target_afdb_dir, rec_id)
        x_rec, y_rec = process_afdb_record(rec_path)
        if len(x_rec) > 0:
            all_X.extend(x_rec)
            all_y.extend(y_rec)
            all_subs.extend([rec_id] * len(y_rec))
            print(f"  Rekaman AFDB {rec_id}: {len(x_rec):,} jendela (AFib: {sum(y_rec):,}, Non-AF: {len(y_rec)-sum(y_rec):,})")
            
    X_arr = np.array(all_X, dtype=np.float32).reshape((-1, WINDOW_SIZE, 1))
    y_arr = np.array(all_y, dtype=np.int32)
    subs_arr = np.array(all_subs, dtype=object)
    
    pred_probs = model.predict(X_arr, batch_size=512, verbose=0).ravel()
    
    print("\n=======================================================================")
    print(" HASIL EVALUASI VALIDASI EKSTERNAL MIT-BIH AFDB (ZERO-SHOT)")
    print("=======================================================================")
    
    eval_thresholds = [
        ('default_0.50', 0.50),
        ('calibrated_cv_T', calibrated_th),
        ('operational_0.85', 0.85)
    ]
    
    ext_results = []
    patient_ext_list = []
    
    for th_name, th in eval_thresholds:
        preds = (pred_probs >= th).astype(int)
        acc = accuracy_score(y_arr, preds)
        sens = recall_score(y_arr, preds, zero_division=0)
        spec = np.sum((y_arr == 0) & (preds == 0)) / np.sum(y_arr == 0) if np.sum(y_arr == 0) > 0 else 1.0
        prec = precision_score(y_arr, preds, zero_division=0)
        f1 = f1_score(y_arr, preds, zero_division=0)
        auc_val = roc_auc_score(y_arr, pred_probs)
        
        print(f"[{th_name} (T={th:.3f})] Acc: {acc*100:.2f}% | Sens: {sens*100:.2f}% | Spec: {spec*100:.2f}% | Prec: {prec*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc_val:.4f}")
        
        ext_results.append({
            'threshold_mode': th_name,
            'threshold_val': th,
            'accuracy': float(round(acc, 4)),
            'sensitivity': float(round(sens, 4)),
            'specificity': float(round(spec, 4)),
            'precision': float(round(prec, 4)),
            'f1_score': float(round(f1, 4)),
            'roc_auc': float(round(auc_val, 4))
        })
        
        # Metrik per pasien pada threshold optimal terkalibrasi
        if th_name == 'calibrated_cv_T':
            for sub in np.unique(subs_arr):
                m = (subs_arr == sub)
                sub_y = y_arr[m]
                sub_p = preds[m]
                n_pos = int(np.sum(sub_y == 1))
                n_neg = int(np.sum(sub_y == 0))
                
                sub_sens = (np.sum((sub_y == 1) & (sub_p == 1)) / n_pos) if n_pos > 0 else np.nan
                sub_spec = (np.sum((sub_y == 0) & (sub_p == 0)) / n_neg) if n_neg > 0 else np.nan
                sub_prec = (np.sum((sub_y == 1) & (sub_p == 1)) / np.sum(sub_p == 1)) if np.sum(sub_p == 1) > 0 else np.nan
                sub_f1 = (2 * sub_prec * sub_sens / (sub_prec + sub_sens)) if (not np.isnan(sub_sens) and not np.isnan(sub_prec) and (sub_sens + sub_prec) > 0) else np.nan
                
                patient_ext_list.append({
                    'subject': str(sub),
                    'total_windows': len(sub_y),
                    'n_af_windows': n_pos,
                    'n_non_af_windows': n_neg,
                    'accuracy': float(round(accuracy_score(sub_y, sub_p), 4)),
                    'sensitivity': float(round(sub_sens, 4)) if not np.isnan(sub_sens) else np.nan,
                    'specificity': float(round(sub_spec, 4)) if not np.isnan(sub_spec) else np.nan,
                    'precision': float(round(sub_prec, 4)) if not np.isnan(sub_prec) else np.nan,
                    'f1_score': float(round(sub_f1, 4)) if not np.isnan(sub_f1) else np.nan
                })
                
    df_ext = pd.DataFrame(ext_results)
    df_ext.to_csv(os.path.join(RESULTS_DIR, 'external_validation_afdb_summary.csv'), index=False)
    
    df_ext_patients = pd.DataFrame(patient_ext_list)
    df_ext_patients.to_csv(os.path.join(RESULTS_DIR, 'external_validation_afdb_per_patient.csv'), index=False)
    
    # Bootstrap 95% CI Tingkat Pasien pada Dataset Eksternal AFDB
    ext_ci_stats = compute_bootstrap_ci(df_ext_patients, random_seed=0)
    with open(os.path.join(RESULTS_DIR, 'external_validation_afdb_bootstrap_ci.json'), 'w', encoding='utf-8') as f:
        json.dump(ext_ci_stats, f, indent=4)
        
    print(f"\n[SUKSES] Laporan validasi eksternal tersimpan di:")
    print(f"  - {RESULTS_DIR}/external_validation_afdb_summary.csv")
    print(f"  - {RESULTS_DIR}/external_validation_afdb_per_patient.csv")
    print(f"  - {RESULTS_DIR}/external_validation_afdb_bootstrap_ci.json")

if __name__ == '__main__':
    run_external_validation()
