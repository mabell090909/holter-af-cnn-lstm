import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

# ==============================================================================
# KONFIGURASI THREADING & DETERMINISME OPERASI
# ==============================================================================
tf.config.threading.set_intra_op_parallelism_threads(18)
tf.config.threading.set_inter_op_parallelism_threads(4)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_PATH = os.path.join(BASE_DIR, 'cnn_lstm_model.keras')
MODEL_PATH_H5 = os.path.join(BASE_DIR, 'cnn_lstm_model.h5')
METADATA_PATH = os.path.join(RESULTS_DIR, 'final_model_metadata.json')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# 1. GENERATOR SEQUENCE UNTUK PELATIHAN BEBAS OUT-OF-MEMORY (OOM)
# ==============================================================================
class ECGMemoryMappedSequence(Sequence):
    """
    Generator Keras Sequence berbasis disk memmap untuk dataset skala besar:
    - Tidak menyalin seluruh data ke RAM saat boolean masking.
    - Hanya memuat potongan batch (512 sampel) secara on-the-fly dari disk.
    - Menghindari OOM pada 84+ rekaman 24 jam (~gigabytes).
    """
    def __init__(self, x_memmap_path, indices, y, batch_size=512, shuffle=True, seed=42):
        super().__init__()
        self.x_memmap_path = x_memmap_path
        self.indices = np.array(indices, dtype=np.int64)
        self.y = np.array(y, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.X_mmap = None
        self.on_epoch_end()
        
    def _open_mmap(self):
        if self.X_mmap is None:
            self.X_mmap = np.load(self.x_memmap_path, mmap_mode='r')
            
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
        
    def __getitem__(self, idx):
        self._open_mmap()
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        # Urutkan index untuk efisiensi seek I/O berurutan pada disk
        sorted_order = np.argsort(batch_indices)
        sorted_idx = batch_indices[sorted_order]
        
        batch_x_sorted = self.X_mmap[sorted_idx]
        batch_x = np.empty_like(batch_x_sorted)
        batch_x[sorted_order] = batch_x_sorted
        
        batch_y = self.y[batch_indices]
        return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.int32)
        
    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

# ==============================================================================
# 2. ARSITEKTUR HYBRID CNN-LSTM
# ==============================================================================
def build_cnn_lstm_model(input_shape=(1280, 1)):
    """
    Arsitektur Hybrid CNN-LSTM untuk Sinyal EKG 1D:
    - 2 Blok Conv1D: Ekstraksi fitur visual morfologi gelombang P-QRS-T
    - 2 Lapisan LSTM: Menangkap dependensi temporal dan iregularitas R-R interval
    - Dense + Dropout + L2: Klasifikasi probabilitas AFib dengan regularisasi ketat
    """
    model = Sequential([
        # Blok Konvolusi 1
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        
        # Blok Konvolusi 2
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        
        # Blok Recurrent (LSTM)
        LSTM(64, return_sequences=True),
        LSTM(32),
        
        # Lapisan Dense & Regularisasi
        Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Probabilitas kontinu P(AFib) in [0, 1]
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# ==============================================================================
# 3. EVALUASI TINGKAT PASIEN (DENGAN PENANGANAN KELAS KOSONG MENGGUNAKAN NaN)
# ==============================================================================
def compute_patient_level_metrics(y_true, y_pred, subjects_test):
    """
    Menghitung metrik per subjek independen (bukan sekadar pooled windows).
    Jika pasien tidak memiliki jendela AF, sensitivitas = np.nan.
    Jika pasien tidak memiliki jendela Non-AF, spesifisitas = np.nan.
    Jika tidak ada prediksi positif (pred_pos == 0), presisi = np.nan (bukan 1.0).
    Ini mencegah distorsi rata-rata akibat pemaksaan nilai semu.
    """
    unique_test_subs = np.unique(subjects_test)
    patient_stats = []
    
    for sub in unique_test_subs:
        mask = (subjects_test == sub)
        sub_y_true = y_true[mask]
        sub_y_pred = y_pred[mask]
        
        n_windows = len(sub_y_true)
        n_pos = int(np.sum(sub_y_true == 1))
        n_neg = int(np.sum(sub_y_true == 0))
        
        true_af_burden = (n_pos / n_windows) * 100.0
        pred_af_burden = (np.sum(sub_y_pred == 1) / n_windows) * 100.0
        burden_error = abs(pred_af_burden - true_af_burden)
        
        sub_acc = accuracy_score(sub_y_true, sub_y_pred)
        
        # Sensitivitas (Recall AFib)
        if n_pos > 0:
            tp_sub = np.sum((sub_y_true == 1) & (sub_y_pred == 1))
            sub_sens = float(tp_sub / n_pos)
        else:
            sub_sens = np.nan
            
        # Spesifisitas (Recall Non-AF)
        if n_neg > 0:
            tn_sub = np.sum((sub_y_true == 0) & (sub_y_pred == 0))
            sub_spec = float(tn_sub / n_neg)
        else:
            sub_spec = np.nan
            
        # Presisi: Jika tidak ada jendela yang diprediksi positif, presisi tidak terdefinisi (np.nan)
        pred_pos = int(np.sum(sub_y_pred == 1))
        if pred_pos > 0:
            tp_sub = np.sum((sub_y_true == 1) & (sub_y_pred == 1))
            sub_prec = float(tp_sub / pred_pos)
        else:
            sub_prec = np.nan
            
        # F1-Score: Terdefinisi hanya jika sens dan prec terdefinisi
        if not np.isnan(sub_sens) and not np.isnan(sub_prec) and (sub_sens + sub_prec) > 0:
            sub_f1 = float(2 * (sub_prec * sub_sens) / (sub_prec + sub_sens))
        else:
            sub_f1 = np.nan
            
        patient_stats.append({
            'subject': str(sub),
            'total_windows': int(n_windows),
            'n_af_windows': n_pos,
            'n_non_af_windows': n_neg,
            'true_af_burden_pct': float(round(true_af_burden, 2)),
            'pred_af_burden_pct': float(round(pred_af_burden, 2)),
            'burden_error_pct': float(round(burden_error, 2)),
            'accuracy': float(round(sub_acc, 4)),
            'sensitivity': float(round(sub_sens, 4)) if not np.isnan(sub_sens) else np.nan,
            'specificity': float(round(sub_spec, 4)) if not np.isnan(sub_spec) else np.nan,
            'precision': float(round(sub_prec, 4)) if not np.isnan(sub_prec) else np.nan,
            'f1_score': float(round(sub_f1, 4)) if not np.isnan(sub_f1) else np.nan
        })
        
    return pd.DataFrame(patient_stats)

def compute_bootstrap_ci(df_patients, n_bootstrap=1000, alpha=0.05, random_seed=0):
    """
    Menghitung 95% Confidence Interval tingkat pasien menggunakan Bootstrap resampling ter-seed (reproducible).
    Menggunakan np.nanmean untuk menangani nilai nan tanpa distorsi.
    """
    rng = np.random.default_rng(random_seed)
    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'burden_error_pct']
    ci_results = {}
    n_patients = len(df_patients)
    
    if n_patients < 3:
        return {}
        
    for m in metrics:
        values = df_patients[m].values
        valid_values = values[~np.isnan(values)]
        n_valid = len(valid_values)
        
        if n_valid < 2:
            mean_val = float(np.nanmean(values))
            ci_results[m] = {
                'mean': float(round(mean_val, 4)),
                'std': 0.0,
                'ci_lower_95': float(round(mean_val, 4)),
                'ci_upper_95': float(round(mean_val, 4)),
                'n_evaluated_patients': n_valid
            }
            continue
            
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(valid_values, size=n_valid, replace=True)
            boot_means.append(np.mean(sample))
            
        low_p = (alpha / 2.0) * 100
        high_p = (1.0 - alpha / 2.0) * 100
        ci_lower = np.percentile(boot_means, low_p)
        ci_upper = np.percentile(boot_means, high_p)
        
        ci_results[m] = {
            'mean': float(round(np.mean(valid_values), 4)),
            'std': float(round(np.std(valid_values), 4)),
            'ci_lower_95': float(round(ci_lower, 4)),
            'ci_upper_95': float(round(ci_upper, 4)),
            'n_evaluated_patients': n_valid
        }
    return ci_results

# ==============================================================================
# 4. PIPELINE ESTIMASI KINERJA (5-FOLD STRATIFIED GROUP CV) & MODEL FINAL
# ==============================================================================
def run_rigorous_training(seeds=[42, 101, 2024], n_splits=5, epochs=15):
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    print("=======================================================================", flush=True)
    print(" Memulai Pelatihan Bebas Leakage & Evaluasi Standar Peer-Review Ilmiah", flush=True)
    print(f" Skema: {n_splits}-Fold Stratified Group K-Fold (Pasien-Level Partitioning)", flush=True)
    print(f" Multi-Seed Repetition: {seeds} untuk Estimasi Robust Mean +- SD", flush=True)
    print(" Arsitektur Data  : Keras Sequence Memmap (Bebas Kehabisan RAM/OOM)", flush=True)
    print(" Model Deployment : Dilatih Ulang Pasca-CV Menggunakan Median Epoch & Threshold", flush=True)
    print("=======================================================================", flush=True)
    
    x_file = os.path.join(PROCESSED_DIR, 'X.npy')
    rr_file = os.path.join(PROCESSED_DIR, 'rr_features.npy')
    y_file = os.path.join(PROCESSED_DIR, 'y.npy')
    sub_file = os.path.join(PROCESSED_DIR, 'subjects.npy')
    
    if not (os.path.exists(x_file) and os.path.exists(y_file) and os.path.exists(sub_file)):
        print(f"[ERROR] Berkas data tensor tidak ditemukan di {PROCESSED_DIR}!")
        return
        
    # Membaca header tensor & metadata via memmap
    X_mmap_info = np.load(x_file, mmap_mode='r')
    total_samples, window_size, channels = X_mmap_info.shape
    y = np.load(y_file)
    subjects = np.load(sub_file, allow_pickle=True)
    
    has_rr_features = os.path.exists(rr_file)
    if has_rr_features:
        rr_features = np.load(rr_file)
        print(f"[DATASET] Matriks Fitur Pan-Tompkins-Style RR termuat: {rr_features.shape}", flush=True)
    else:
        rr_features = None
        
    unique_subs = np.unique(subjects)
    n_unique_subs = len(unique_subs)
    
    print(f"\n[DATASET] Total Jendela EKG 10-Detik : {total_samples:,}", flush=True)
    print(f"[DATASET] Total Subjek / Pasien Unik: {n_unique_subs}", flush=True)
    print(f"[DATASET] Kelas 0 (Non-AF)          : {np.sum(y == 0):,} ({(np.sum(y == 0)/len(y))*100:.2f}%)", flush=True)
    print(f"[DATASET] Kelas 1 (AFib)            : {np.sum(y == 1):,} ({(np.sum(y == 1)/len(y))*100:.2f}%)", flush=True)
    
    fold_audit = {}
    all_fold_metrics = []
    patient_dfs_optimal_T = []
    all_rf_fold_metrics = []
    all_rf_patient_dfs = []
    
    cv_best_epochs = []
    cv_optimal_thresholds = []
    
    for s_idx, current_seed in enumerate(seeds, start=1):
        print(f"\n#######################################################################", flush=True)
        print(f" >>> RUNNING EXPERIMENT REPETITION: SEED {current_seed} ({s_idx}/{len(seeds)})", flush=True)
        print(f"#######################################################################", flush=True)
        
        tf.keras.utils.set_random_seed(current_seed)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
        all_indices = np.arange(total_samples)
        
        for fold, (train_idx, test_idx) in enumerate(sgkf.split(all_indices, y, groups=subjects), start=1):
            tf.keras.utils.set_random_seed(current_seed * 100 + fold)
            test_subs_fold = np.unique(subjects[test_idx])
            train_subs_fold = np.unique(subjects[train_idx])
            
            # Verifikasi Zero Data Leakage Antar Pasien
            leakage = set(train_subs_fold).intersection(set(test_subs_fold))
            assert len(leakage) == 0, f"[FATAL] Data leakage terdeteksi di Seed {current_seed} Fold {fold}: {leakage}"
            
            # Pembagian Inner Validation Terstratifikasi:
            subs_with_af = [s for s in train_subs_fold if np.sum((subjects == s) & (y == 1)) > 0]
            subs_without_af = [s for s in train_subs_fold if np.sum((subjects == s) & (y == 1)) == 0]
            
            rng = np.random.RandomState(current_seed * 100 + fold)
            rng.shuffle(subs_with_af)
            rng.shuffle(subs_without_af)
            
            val_subs_inner = []
            if len(subs_with_af) > 0:
                val_subs_inner.append(subs_with_af[0])
            if len(subs_without_af) > 0:
                val_subs_inner.append(subs_without_af[0])
                
            target_val_count = max(2, int(len(train_subs_fold) * 0.2))
            remaining_train = [s for s in train_subs_fold if s not in val_subs_inner]
            rng.shuffle(remaining_train)
            while len(val_subs_inner) < target_val_count and len(remaining_train) > 0:
                val_subs_inner.append(remaining_train.pop())
                
            train_subs_inner = [s for s in train_subs_fold if s not in val_subs_inner]
            
            train_inner_mask = np.isin(subjects, train_subs_inner)
            val_inner_mask = np.isin(subjects, val_subs_inner)
            test_mask = np.isin(subjects, test_subs_fold)
            
            train_indices = all_indices[train_inner_mask]
            val_indices = all_indices[val_inner_mask]
            test_indices = all_indices[test_mask]
            
            y_tr = y[train_indices]
            y_val = y[val_indices]
            y_test = y[test_indices]
            subs_test = subjects[test_indices]
            
            # Validasi bahwa kedua kelas tersedia di val inner
            assert y_val.sum() > 0 and (y_val == 0).sum() > 0, f"Val tanpa dua kelas: {val_subs_inner}"
            
            split_key = f"seed_{current_seed}_fold_{fold}"
            fold_audit[split_key] = {
                'seed': current_seed,
                'fold': fold,
                'train_subjects': train_subs_inner,
                'val_inner_subjects': val_subs_inner,
                'test_subjects': test_subs_fold.tolist(),
                'train_windows': int(len(train_indices)),
                'val_windows': int(len(val_indices)),
                'test_windows': int(len(test_indices))
            }
            
            fold_ckpt_path = os.path.join(RESULTS_DIR, f"fold_checkpoint_seed_{current_seed}_f{fold}.json")
            if os.path.exists(fold_ckpt_path):
                try:
                    with open(fold_ckpt_path, 'r', encoding='utf-8') as f:
                        ckpt = json.load(f)
                    ckpt_samples = ckpt.get('dataset_total_samples')
                    ckpt_patients = ckpt.get('dataset_unique_patients')
                    if ckpt_samples == int(total_samples) and ckpt_patients == int(n_unique_subs):
                        print(f"\n[RESUME] Seed {current_seed} Fold {fold} cocok ({ckpt_samples:,} sampel, {ckpt_patients} pasien). Memuat hasil dan melewati fold ini...", flush=True)
                        cv_best_epochs.append(ckpt['best_epoch'])
                        cv_optimal_thresholds.append(ckpt['best_th'])
                        all_fold_metrics.extend(ckpt['all_fold_metrics'])
                        if ckpt.get('patient_metrics_optimal_T'):
                            patient_dfs_optimal_T.append(pd.DataFrame(ckpt['patient_metrics_optimal_T']))
                        if ckpt.get('rf_fold_metrics'):
                            all_rf_fold_metrics.append(ckpt['rf_fold_metrics'])
                        if ckpt.get('rf_patient_metrics'):
                            all_rf_patient_dfs.append(pd.DataFrame(ckpt['rf_patient_metrics']))
                        continue
                    else:
                        print(f"\n[PERINGATAN RESUME] Checkpoint {fold_ckpt_path} tidak cocok ({ckpt_samples} vs {total_samples} sampel, {ckpt_patients} vs {n_unique_subs} pasien). Menjalankan ulang fold...", flush=True)
                except Exception as e:
                    print(f"\n[PERINGATAN RESUME] Gagal membaca checkpoint {fold_ckpt_path}: {e}. Menjalankan ulang fold...", flush=True)
                
            print(f"\n--- [Seed {current_seed}] FOLD {fold}/{n_splits} ---", flush=True)
            print(f" Train Subs ({len(train_subs_inner)}): {train_subs_inner}", flush=True)
            print(f" Val Subs   ({len(val_subs_inner)}): {val_subs_inner}", flush=True)
            print(f" Test Subs  ({len(test_subs_fold)}): {test_subs_fold.tolist()}", flush=True)
            
            classes = np.unique(y_tr)
            weights = compute_class_weight('balanced', classes=classes, y=y_tr)
            class_weight_dict = {c: float(w) for c, w in zip(classes, weights)}
            
            # ------------------------------------------------------------------
            # A. Baseline Pembanding Simetris: Random Forest (Pan-Tompkins RR Features)
            # ------------------------------------------------------------------
            if has_rr_features:
                rf_tr_X = rr_features[train_indices]
                rf_val_X = rr_features[val_indices]
                rf_te_X = rr_features[test_indices]
                
                rf_baseline = RandomForestClassifier(
                    n_estimators=100, max_depth=10, class_weight='balanced',
                    random_state=current_seed, n_jobs=-1
                )
                rf_baseline.fit(rf_tr_X, y_tr)
                
                rf_val_prob = rf_baseline.predict_proba(rf_val_X)[:, 1] if len(rf_baseline.classes_) > 1 else np.zeros(len(y_val))
                rf_test_prob = rf_baseline.predict_proba(rf_te_X)[:, 1] if len(rf_baseline.classes_) > 1 else np.zeros(len(y_test))
                
                # Tuning threshold RF pada val_inner (Simetris dengan CNN-LSTM)
                cand_th = np.linspace(0.2, 0.85, 20)
                best_rf_th, best_rf_val_f1 = 0.5, -1.0
                for th in cand_th:
                    p = (rf_val_prob >= th).astype(int)
                    f = f1_score(y_val, p, zero_division=0)
                    if f > best_rf_val_f1:
                        best_rf_val_f1 = f
                        best_rf_th = float(th)
                        
                rf_pred_opt = (rf_test_prob >= best_rf_th).astype(int)
                rf_pred_050 = (rf_test_prob >= 0.50).astype(int)
                
                rf_acc_opt = accuracy_score(y_test, rf_pred_opt)
                rf_f1_opt = f1_score(y_test, rf_pred_opt, zero_division=0)
                rf_auc = roc_auc_score(y_test, rf_test_prob) if len(np.unique(y_test)) > 1 else 0.5
                
                all_rf_fold_metrics.append({
                    'seed': current_seed,
                    'fold': fold,
                    'rf_optimal_threshold': best_rf_th,
                    'rf_accuracy_T_opt': rf_acc_opt,
                    'rf_f1_T_opt': rf_f1_opt,
                    'rf_accuracy_T050': accuracy_score(y_test, rf_pred_050),
                    'rf_f1_T050': f1_score(y_test, rf_pred_050, zero_division=0),
                    'rf_roc_auc': rf_auc
                })
                print(f"[Seed {current_seed} Fold {fold}] Baseline RF: Acc (T*={best_rf_th:.2f}) = {rf_acc_opt*100:.2f}%, F1 = {rf_f1_opt*100:.2f}%, AUC = {rf_auc:.4f}", flush=True)
                
                # Simpan metrik per pasien RF untuk uji Wilcoxon
                df_rf_patients = compute_patient_level_metrics(y_test, rf_pred_opt, subs_test)
                df_rf_patients['seed'] = current_seed
                df_rf_patients['fold'] = fold
                all_rf_patient_dfs.append(df_rf_patients)
                
            # ------------------------------------------------------------------
            # B. Pelatihan Model CNN-LSTM via Keras Sequence (Zero-RAM Overhead)
            # ------------------------------------------------------------------
            train_gen = ECGMemoryMappedSequence(x_file, train_indices, y, batch_size=512, shuffle=True, seed=current_seed*100+fold)
            val_gen = ECGMemoryMappedSequence(x_file, val_indices, y, batch_size=512, shuffle=False)
            test_gen = ECGMemoryMappedSequence(x_file, test_indices, y, batch_size=512, shuffle=False)
            
            model = build_cnn_lstm_model(input_shape=(window_size, channels))
            fold_model_path = os.path.join(RESULTS_DIR, f"best_model_seed_{current_seed}_fold_{fold}.h5")
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=0)
            callbacks = [
                ModelCheckpoint(fold_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0),
                early_stop,
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)
            ]
            
            print(f"[Seed {current_seed} Fold {fold}] Melatih CNN-LSTM via Keras Sequence ({len(train_indices):,} segmen latih)...", flush=True)
            history = model.fit(
                train_gen,
                validation_data=val_gen, # <-- HANYA VALIDASI INTERNAL
                epochs=epochs,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            if os.path.exists(fold_model_path):
                model = tf.keras.models.load_model(fold_model_path)
                
            best_epoch = int(early_stop.best_epoch + 1) if (hasattr(early_stop, 'best_epoch') and early_stop.best_epoch is not None) else len(history.history['loss'])
            cv_best_epochs.append(best_epoch)
            
            # ------------------------------------------------------------------
            # C. Penentuan Optimal Threshold dari Val Inner (Bukan Test Set)
            # ------------------------------------------------------------------
            val_pred_probs = model.predict(val_gen, verbose=0).ravel()
            candidate_thresholds = np.linspace(0.2, 0.85, 20)
            best_th, best_val_f1 = 0.5, -1.0
            for th in candidate_thresholds:
                v_pred = (val_pred_probs >= th).astype(int)
                v_f1 = f1_score(y_val, v_pred, zero_division=0)
                if v_f1 > best_val_f1:
                    best_val_f1 = v_f1
                    best_th = float(th)
                    
            cv_optimal_thresholds.append(best_th)
            print(f"[Seed {current_seed} Fold {fold}] Best Epoch: {best_epoch} | Optimal Threshold Val: T* = {best_th:.3f} (Val F1: {best_val_f1*100:.2f}%)", flush=True)
            
            # ------------------------------------------------------------------
            # D. Evaluasi Akhir pada Test Set Terisolasi (Estimasi CV)
            # ------------------------------------------------------------------
            test_pred_probs = model.predict(test_gen, verbose=0).ravel()
            
            for th_name, th_val in [('default_0.50', 0.50), ('optimal_val_T', best_th), ('operational_0.85', 0.85)]:
                y_pred = (test_pred_probs >= th_val).astype(int)
                
                cm = confusion_matrix(y_test, y_pred)
                tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
                fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
                fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
                tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0
                
                acc = accuracy_score(y_test, y_pred)
                sens = recall_score(y_test, y_pred, zero_division=0)
                spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
                prec = precision_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, test_pred_probs) if len(np.unique(y_test)) > 1 else 0.5
                
                fold_res = {
                    'seed': current_seed,
                    'fold': fold,
                    'threshold_mode': th_name,
                    'threshold_val': float(round(th_val, 4)),
                    'accuracy': float(acc),
                    'sensitivity': float(sens),
                    'specificity': float(spec),
                    'precision': float(prec),
                    'f1_score': float(f1),
                    'roc_auc': float(roc_auc),
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                    'test_windows': len(y_test)
                }
                all_fold_metrics.append(fold_res)
                
                if th_name == 'optimal_val_T':
                    df_pat_opt = compute_patient_level_metrics(y_test, y_pred, subs_test)
                    df_pat_opt['seed'] = current_seed
                    df_pat_opt['fold'] = fold
                    df_pat_opt['threshold_used'] = best_th
                    patient_dfs_optimal_T.append(df_pat_opt)
                    print(f"[Seed {current_seed} Fold {fold} Hasil T*={best_th:.3f}] Acc: {acc*100:.2f}% | Sens: {sens*100:.2f}% | Spec: {spec*100:.2f}% | F1: {f1*100:.2f}% | AUC: {roc_auc:.4f}", flush=True)
                    
            auc_rf_fold = all_rf_fold_metrics[-1]['rf_roc_auc'] if len(all_rf_fold_metrics) > 0 else 0.5
            print(f"[Seed {current_seed} Fold {fold}] epoch={best_epoch} T*={best_th:.2f} | "
                  f"AUC CNN={all_fold_metrics[-1]['roc_auc']:.4f} | AUC RF={auc_rf_fold:.4f}", flush=True)
                  
            # ------------------------------------------------------------------
            # E. Simpan Checkpoint Per Fold (Syarat 1 Quality Gate)
            # ------------------------------------------------------------------
            fold_ckpt_data = {
                'seed': current_seed,
                'fold': fold,
                'dataset_total_samples': int(total_samples),
                'dataset_unique_patients': int(n_unique_subs),
                'best_epoch': int(best_epoch),
                'best_th': float(best_th),
                'best_val_f1': float(best_val_f1),
                'all_fold_metrics': [m for m in all_fold_metrics if m['seed'] == current_seed and m['fold'] == fold],
                'patient_metrics_optimal_T': df_pat_opt.to_dict(orient='records') if ('df_pat_opt' in locals() and df_pat_opt is not None) else [],
                'rf_fold_metrics': all_rf_fold_metrics[-1] if (has_rr_features and len(all_rf_fold_metrics) > 0) else {},
                'rf_patient_metrics': df_rf_patients.to_dict(orient='records') if (has_rr_features and 'df_rf_patients' in locals() and df_rf_patients is not None) else []
            }
            with open(fold_ckpt_path, 'w', encoding='utf-8') as f:
                json.dump(fold_ckpt_data, f, indent=4, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x.tolist() if isinstance(x, np.ndarray) else str(x))
            print(f"[Seed {current_seed} Fold {fold}] Checkpoint berhasil disimpan ke: {fold_ckpt_path}", flush=True)
                    
    # ==============================================================================
    # 5. PELATIHAN MODEL FINAL INDEPENDEN (TANPA MODEL SELECTION LEAKAGE)
    # ==============================================================================
    median_best_epoch = int(round(float(np.median(cv_best_epochs))))
    median_best_th = float(round(float(np.median(cv_optimal_thresholds)), 4))
    
    print("\n=======================================================================", flush=True)
    print(" MELATIH MODEL FINAL INDEPENDEN UNTUK DEPLOYMENT & VALIDASI EKSTERNAL", flush=True)
    print(f" Calibrated Best Epoch     : {median_best_epoch} (Median dari Cross-Validation)", flush=True)
    print(f" Calibrated Optimal Thresh : {median_best_th:.4f} (Median T* dari Inner Val)", flush=True)
    print("=======================================================================", flush=True)
    
    final_meta = {
        'median_best_epoch': median_best_epoch,
        'median_optimal_threshold': median_best_th,
        'cv_best_epochs': cv_best_epochs,
        'cv_optimal_thresholds': cv_optimal_thresholds
    }

    if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
        print(f"\n[RESUME] Model final sudah ada di {MODEL_PATH}. Memuat metadata...", flush=True)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            final_meta = json.load(f)
    else:
        classes_all = np.unique(y)
        weights_all = compute_class_weight('balanced', classes=classes_all, y=y)
        cw_all_dict = {c: float(w) for c, w in zip(classes_all, weights_all)}
        
        full_train_gen = ECGMemoryMappedSequence(x_file, all_indices, y, batch_size=512, shuffle=True, seed=seeds[0])
        
        tf.keras.utils.set_random_seed(seeds[0])
        final_model = build_cnn_lstm_model(input_shape=(window_size, channels))
        final_model.fit(
            full_train_gen,
            epochs=median_best_epoch,
            class_weight=cw_all_dict,
            verbose=1
        )
        final_model.save(MODEL_PATH)
        try:
            final_model.save(MODEL_PATH_H5)
        except Exception:
            pass
        print(f"[SUKSES] Model final tersimpan di: {MODEL_PATH} dan {MODEL_PATH_H5}", flush=True)
        
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_meta, f, indent=4)
        
    # ==============================================================================
    # 6. PENYUSUNAN LAPORAN ILMIAH & STATISTIK AGREGAT
    # ==============================================================================
    print("\n=======================================================================", flush=True)
    print(" RANGKUMAN HASIL AKHIR (AGREGAT STRATIFIED GROUP CV)", flush=True)
    print("=======================================================================", flush=True)
    
    df_metrics = pd.DataFrame(all_fold_metrics)
    df_metrics.to_csv(os.path.join(RESULTS_DIR, 'cv_5fold_metrics_by_threshold.csv'), index=False)
    
    summary_by_th = df_metrics.groupby('threshold_mode').agg({
        'accuracy': ['mean', 'std'],
        'sensitivity': ['mean', 'std'],
        'specificity': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'f1_score': ['mean', 'std'],
        'roc_auc': ['mean', 'std']
    }).round(4)
    print("\nRingkasan Performa Lintas Fold & Seed (Mean +- SD):")
    print(summary_by_th)
    
    metric_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'burden_error_pct']
    if len(patient_dfs_optimal_T) > 0:
        df_all_patients_opt = pd.concat(patient_dfs_optimal_T, ignore_index=True)
        df_all_patients_opt.to_csv(os.path.join(RESULTS_DIR, 'patient_level_metrics_all_subjects.csv'), index=False)
        df_all_patients_opt.to_csv(os.path.join(RESULTS_DIR, 'patient_level_metrics_optimal_threshold.csv'), index=False)
        
        df_pat_mean = df_all_patients_opt.groupby('subject')[metric_cols].mean().reset_index()
        ci_stats_opt = compute_bootstrap_ci(df_pat_mean, random_seed=0)
        
        print("\nEstimasi Metrik Tingkat Pasien pada Optimal Threshold T* (Bootstrap 95% CI):")
        for k, v in ci_stats_opt.items():
            if k == 'burden_error_pct':
                print(f"  {k:18s}: {v['mean']:.2f}% +- {v['std']:.2f}% [95% CI: {v['ci_lower_95']:.2f}% - {v['ci_upper_95']:.2f}%] (N={v['n_evaluated_patients']})")
            else:
                print(f"  {k:18s}: {v['mean']*100:.2f}% +- {v['std']*100:.2f}% [95% CI: {v['ci_lower_95']*100:.2f}% - {v['ci_upper_95']*100:.2f}%] (N={v['n_evaluated_patients']})")
    else:
        ci_stats_opt = {}
        
    # Baseline comparison summary
    if len(all_rf_fold_metrics) > 0:
        df_rf_metrics = pd.DataFrame(all_rf_fold_metrics)
        df_rf_metrics.to_csv(os.path.join(RESULTS_DIR, 'baseline_rf_comparison.csv'), index=False)
        
        df_all_rf_patients = pd.concat(all_rf_patient_dfs, ignore_index=True)
        df_all_rf_patients.to_csv(os.path.join(RESULTS_DIR, 'patient_level_metrics_rf.csv'), index=False)
        
        print("\nPerbandingan Baseline Simetris (Random Forest vs CNN-LSTM):")
        print(f"  RF Accuracy (T*)   : {df_rf_metrics['rf_accuracy_T_opt'].mean()*100:.2f}% +- {df_rf_metrics['rf_accuracy_T_opt'].std()*100:.2f}%")
        print(f"  RF F1-Score (T*)   : {df_rf_metrics['rf_f1_T_opt'].mean()*100:.2f}% +- {df_rf_metrics['rf_f1_T_opt'].std()*100:.2f}%")
        print(f"  RF ROC-AUC         : {df_rf_metrics['rf_roc_auc'].mean():.4f} +- {df_rf_metrics['rf_roc_auc'].std():.4f}")
        
        f1_mean_cnn = df_metrics[df_metrics['threshold_mode']=='optimal_val_T']['f1_score'].mean()
        f1_std_cnn = df_metrics[df_metrics['threshold_mode']=='optimal_val_T']['f1_score'].std()
        auc_mean_cnn = df_metrics[df_metrics['threshold_mode']=='optimal_val_T']['roc_auc'].mean()
        auc_std_cnn = df_metrics[df_metrics['threshold_mode']=='optimal_val_T']['roc_auc'].std()
        print(f"  CNN-LSTM F1 (T*)   : {f1_mean_cnn*100:.2f}% +- {f1_std_cnn*100:.2f}%")
        print(f"  CNN-LSTM ROC-AUC   : {auc_mean_cnn:.4f} +- {auc_std_cnn:.4f}")
    else:
        df_rf_metrics = pd.DataFrame()
        f1_mean_cnn, auc_mean_cnn = 0.0, 0.0
        
    with open(os.path.join(RESULTS_DIR, 'fold_splits_audit.json'), 'w', encoding='utf-8') as f:
        json.dump(fold_audit, f, indent=4)
        
    unique_ltafdb = [s for s in unique_subs if str(s).startswith('ltafdb')]
    unique_shdb = [s for s in unique_subs if str(s).startswith('shdb')]
    
    summary_by_th_flat = summary_by_th.copy()
    summary_by_th_flat.columns = [f"{col[0]}_{col[1]}" for col in summary_by_th.columns]
    summary_dict = summary_by_th_flat.to_dict(orient='index')
    
    final_audit_summary = {
        'evaluation_protocol': f'{n_splits}-Fold Stratified Group K-Fold Cross-Validation (Zero Subject Leakage)',
        'data_sources': {
            'ltafdb': f'PhysioNet Long-Term Atrial Fibrillation Database ({len(unique_ltafdb)} subjek lengkap)',
            'shdb-af': f'SHDB-AF: A Japanese Holter ECG Database of Atrial Fibrillation ({len(unique_shdb)} subjek lengkap)'
        },
        'total_windows_10s': int(total_samples),
        'total_unique_subjects': int(n_unique_subs),
        'calibrated_final_model': {
            'median_best_epoch': median_best_epoch,
            'median_optimal_threshold': median_best_th
        },
        'cross_validation_metrics_summary': summary_dict,
        'patient_level_bootstrap_95ci_optimal_T': ci_stats_opt,
        'symmetric_baseline_comparison': {
            'baseline_model': 'Random Forest on Pan-Tompkins-Style Continuous RR Features with inner-val threshold tuning & balanced class weights',
            'rf_accuracy_T_opt_mean': float(round(df_rf_metrics['rf_accuracy_T_opt'].mean(), 4)) if len(df_rf_metrics) > 0 else 0.0,
            'rf_f1_T_opt_mean': float(round(df_rf_metrics['rf_f1_T_opt'].mean(), 4)) if len(df_rf_metrics) > 0 else 0.0,
            'rf_roc_auc_mean': float(round(df_rf_metrics['rf_roc_auc'].mean(), 4)) if len(df_rf_metrics) > 0 else 0.0,
            'cnn_lstm_accuracy_T_opt_mean': float(round(df_metrics[df_metrics['threshold_mode']=='optimal_val_T']['accuracy'].mean(), 4)),
            'cnn_lstm_f1_T_opt_mean': float(round(f1_mean_cnn, 4)),
            'cnn_lstm_roc_auc_mean': float(round(auc_mean_cnn, 4))
        }
    }
    with open(os.path.join(RESULTS_DIR, 'rigorous_evaluation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(final_audit_summary, f, indent=4)
        
    print(f"\n[SUKSES] Seluruh berkas audit ilmiah tersimpan di: {RESULTS_DIR}/")

if __name__ == '__main__':
    # Mode Produksi Penuh (Opsi A) Sesuai Standar Audit Ilmiah (Seed 42, 5-Fold Stratified Group CV)
    run_rigorous_training(seeds=[42], n_splits=5, epochs=15)
