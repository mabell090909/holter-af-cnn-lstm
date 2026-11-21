import os
import time
import json
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
from scipy.signal import iirnotch, filtfilt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Threading config
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(4)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
cache_file = os.path.join(RESULTS_DIR, 'ablation_cohort_dataset.npz')

print("==========================================================================", flush=True)
print(" RUNNING 4TH ABLATION CONDITION: BANDPASS + NOTCHES (50/60 Hz) (REVIEWER 2.2)", flush=True)
print("==========================================================================", flush=True)

# 1. Load cached dataset
cohort_data = np.load(cache_file)
X_bp_all = cohort_data['X_bp'] # shape (20000, 1280, 1)
y_raw_all = cohort_data['y_raw']
subjects_raw_all = cohort_data['subjects_raw']
unique_patients = np.unique(subjects_raw_all)

# Categorize patients
pat_af_counts = {p: int(np.sum((subjects_raw_all == p) & (y_raw_all == 1))) for p in unique_patients}
pats_with_af = [p for p in unique_patients if pat_af_counts[p] > 50]
pats_without_af = [p for p in unique_patients if pat_af_counts[p] <= 50]

# 2. Design and apply dual 50 Hz and 60 Hz IIR notch filters on X_bp_all
print("[1/3] Filtering X_bp with 50 Hz & 60 Hz IIR Notches (Q=12.0)...", flush=True)
fs = 128
b50, a50 = iirnotch(w0=50.0, Q=12.0, fs=fs)
b60, a60 = iirnotch(w0=60.0, Q=12.0, fs=fs)

# Filter 20,000 windows
X_notch = np.empty_like(X_bp_all)
for i in range(len(X_bp_all)):
    sig = X_bp_all[i, :, 0]
    sig_n = filtfilt(b60, a60, filtfilt(b50, a50, sig))
    X_notch[i, :, 0] = sig_n

print(f"      Filtered {len(X_notch):,} windows successfully.", flush=True)

def build_model():
    tf.keras.backend.clear_session()
    model = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same', input_shape=(1280, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

SEEDS = [42, 101, 2024]
new_runs = []

print("\n[2/3] Training and evaluating on Seeds [42, 101, 2024]...", flush=True)
for seed in SEEDS:
    start_t = time.time()
    rng = np.random.RandomState(seed)
    shuffled_af = rng.permutation(pats_with_af)
    shuffled_sinus = rng.permutation(pats_without_af)
    
    n_af_test = max(1, int(len(pats_with_af) * 0.30))
    n_sinus_test = max(1, int(len(pats_without_af) * 0.30))
    
    test_patients = list(shuffled_af[:n_af_test]) + list(shuffled_sinus[:n_sinus_test])
    train_patients = list(shuffled_af[n_af_test:]) + list(shuffled_sinus[n_sinus_test:])
    
    val_patients = [train_patients[0], train_patients[-1]]
    train_inner_patients = train_patients[1:-1]
    
    train_mask = np.isin(subjects_raw_all, train_inner_patients)
    val_mask = np.isin(subjects_raw_all, val_patients)
    test_mask = np.isin(subjects_raw_all, test_patients)
    
    X_tr_cand = X_notch[train_mask]
    y_tr_cand = y_raw_all[train_mask]
    subs_tr_cand = subjects_raw_all[train_mask]
    
    # Cap training samples per patient to 200 windows
    tr_idx_list = []
    for p in np.unique(subs_tr_cand):
        p_idxs = np.where(subs_tr_cand == p)[0]
        if len(p_idxs) > 200:
            p_idxs = rng.choice(p_idxs, 200, replace=False)
        tr_idx_list.extend(p_idxs)
        
    X_tr = X_tr_cand[tr_idx_list]
    y_tr = y_tr_cand[tr_idx_list]
    
    X_val = X_notch[val_mask]
    y_val = y_raw_all[val_mask]
    
    X_te = X_notch[test_mask]
    y_te = y_raw_all[test_mask]
    
    tf.keras.utils.set_random_seed(seed)
    model = build_model()
    
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=64,
        verbose=0
    )
    
    val_probs = model.predict(X_val, verbose=0, batch_size=128).ravel()
    best_th, best_val_f1 = 0.5, -1.0
    for th in np.linspace(0.2, 0.85, 20):
        f = f1_score(y_val, (val_probs >= th).astype(int), zero_division=0)
        if f > best_val_f1:
            best_val_f1 = f
            best_th = float(th)
            
    te_probs = model.predict(X_te, verbose=0, batch_size=128).ravel()
    te_pred = (te_probs >= best_th).astype(int)
    
    acc = accuracy_score(y_te, te_pred)
    f1 = f1_score(y_te, te_pred, zero_division=0)
    sens = recall_score(y_te, te_pred, zero_division=0)
    spec = float(np.sum((y_te == 0) & (te_pred == 0)) / max(1, np.sum(y_te == 0)))
    prec = precision_score(y_te, te_pred, zero_division=0)
    auc_score = roc_auc_score(y_te, te_probs) if len(np.unique(y_te)) > 1 else 0.5
    
    elapsed = time.time() - start_t
    print(f"   [Seed {seed}] Bandpass + Notches | T*={best_th:.2f} | AUC: {auc_score:.4f} | F1: {f1*100:.2f}% | Acc: {acc*100:.2f}% | Sens: {sens*100:.2f}% | Spec: {spec*100:.2f}% ({elapsed:.1f}s)", flush=True)
    
    new_runs.append({
        'seed': seed,
        'condition_id': 'C2b_BP_Notches_Unweighted',
        'condition_name': 'Bandpass + Notches (0.5-40 Hz + 50/60 Hz Notches, Unweighted)',
        'filter_stage': 'bandpass_notches',
        'balancing_strategy': 'unweighted',
        'optimal_threshold': round(best_th, 4),
        'roc_auc': round(auc_score, 4),
        'f1_score': round(f1 * 100, 2),
        'accuracy': round(acc * 100, 2),
        'sensitivity': round(sens * 100, 2),
        'specificity': round(spec * 100, 2),
        'precision': round(prec * 100, 2),
        'runtime_sec': round(elapsed, 1)
    })
    
    del model
    gc.collect()

# 3. Merge with existing raw runs
df_existing_raw = pd.read_csv(os.path.join(RESULTS_DIR, 'ablation_study_raw_runs.csv'))
# Remove C2b if already present
df_existing_raw = df_existing_raw[df_existing_raw['condition_id'] != 'C2b_BP_Notches_Unweighted']
df_new = pd.DataFrame(new_runs)

df_all_raw = pd.concat([df_existing_raw, df_new], ignore_index=True)

# Order conditions logically: C1, C2, C2b, C3, C4, C5
COND_ORDER = [
    ('C1_Raw_Unweighted', 'Raw ECG (No DSP Filtering, Unweighted)', 'raw', 'unweighted'),
    ('C2_Bandpass_Unweighted', 'Bandpass-Only (0.5-40 Hz, No QG, Unweighted)', 'bandpass', 'unweighted'),
    ('C2b_BP_Notches_Unweighted', 'Bandpass + Notches (0.5-40 Hz + 50/60 Hz Notches, Unweighted)', 'bandpass_notches', 'unweighted'),
    ('C3_FullDSP_Unweighted', 'Full DSP Cascade (Dual-Stage + QG, Unweighted)', 'full_dsp', 'unweighted'),
    ('C4_FullDSP_SMOTE', 'Full DSP Cascade + SMOTE Oversampling', 'full_dsp', 'smote'),
    ('C5_FullDSP_ClassWeight', 'Full DSP Cascade + Cost-Sensitive Weight (Proposed)', 'full_dsp', 'class_weight')
]

df_all_raw['cond_rank'] = df_all_raw['condition_id'].map({c[0]: i for i, c in enumerate(COND_ORDER)})
df_all_raw = df_all_raw.sort_values(by=['cond_rank', 'seed']).drop(columns=['cond_rank'])
raw_runs_csv = os.path.join(RESULTS_DIR, 'ablation_study_raw_runs.csv')
df_all_raw.to_csv(raw_runs_csv, index=False)

# Compute unified summary
summary_rows = []
for cid, cname, fstage, bstrat in COND_ORDER:
    sub = df_all_raw[df_all_raw['condition_id'] == cid]
    mean_auc, std_auc = sub['roc_auc'].mean(), sub['roc_auc'].std(ddof=1)
    mean_f1, std_f1 = sub['f1_score'].mean(), sub['f1_score'].std(ddof=1)
    mean_acc, std_acc = sub['accuracy'].mean(), sub['accuracy'].std(ddof=1)
    mean_sens, std_sens = sub['sensitivity'].mean(), sub['sensitivity'].std(ddof=1)
    mean_spec, std_spec = sub['specificity'].mean(), sub['specificity'].std(ddof=1)
    mean_prec, std_prec = sub['precision'].mean(), sub['precision'].std(ddof=1)
    mean_th = sub['optimal_threshold'].mean()
    
    summary_rows.append({
        'condition_id': cid,
        'condition_name': cname,
        'filter_stage': fstage,
        'balancing_strategy': bstrat,
        'mean_optimal_threshold': round(mean_th, 4),
        'roc_auc_mean': round(mean_auc, 4),
        'roc_auc_std': round(std_auc, 4),
        'f1_score_mean': round(mean_f1, 2),
        'f1_score_std': round(std_f1, 2),
        'accuracy_mean': round(mean_acc, 2),
        'accuracy_std': round(std_acc, 2),
        'sensitivity_mean': round(mean_sens, 2),
        'sensitivity_std': round(std_sens, 2),
        'specificity_mean': round(mean_spec, 2),
        'specificity_std': round(std_spec, 2),
        'precision_mean': round(mean_prec, 2),
        'precision_std': round(std_prec, 2),
        'formatted_auc': f"{mean_auc:.4f} \u00b1 {std_auc:.4f}",
        'formatted_f1': f"{mean_f1:.2f}% \u00b1 {std_f1:.2f}%",
        'formatted_acc': f"{mean_acc:.2f}% \u00b1 {std_acc:.2f}%",
        'formatted_sens': f"{mean_sens:.2f}% \u00b1 {std_sens:.2f}%",
        'formatted_spec': f"{mean_spec:.2f}% \u00b1 {std_spec:.2f}%",
        'formatted_prec': f"{mean_prec:.2f}% \u00b1 {std_prec:.2f}%"
    })

df_summary = pd.DataFrame(summary_rows)
summary_csv = os.path.join(RESULTS_DIR, 'ablation_study_summary.csv')
df_summary.to_csv(summary_csv, index=False)

summary_json = os.path.join(RESULTS_DIR, 'ablation_study_summary.json')
with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump({
        'study_title': 'Ablation Analysis: 4-Stage Preprocessing DSP Cascade vs Class Imbalance Handling (SMOTE)',
        'shuffles_count': len(SEEDS),
        'seeds_evaluated': SEEDS,
        'patient_leakage_guarantee': 'Zero leakage (Patient-Level Partitioning)',
        'standard_deviation_convention': 'Unbiased sample standard deviation (ddof=1)',
        'summary_results': summary_rows
    }, f, indent=4)

# Markdown table
md_table = """### Table R6: Comprehensive 4-Stage Preprocessing & Imbalance Mitigation Ablation Study
*Empirical evaluation across 3 independent random patient-level shuffles with zero patient leakage. Results reported as Mean \u00b1 SD (sample standard deviation, ddof=1).*

| Preprocessing Stage | Balancing Strategy | Calibrated $T^*$ | ROC-AUC | F1-Score (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
for r in summary_rows:
    p_name = r['condition_name'].split('(')[0].strip()
    md_table += f"| **{p_name}** | {r['balancing_strategy'].replace('_', ' ').title()} | {r['mean_optimal_threshold']:.2f} | **{r['formatted_auc']}** | {r['formatted_f1']} | {r['formatted_acc']} | {r['formatted_sens']} | {r['formatted_spec']} | {r['formatted_prec']} |\n"

md_path = os.path.join(RESULTS_DIR, 'ablation_study_markdown_table.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_table)

print("\n" + md_table)
print("\n[SUCCESS] 4th Ablation Stage (Bandpass + Notches) successfully integrated into summary tables and JSON!", flush=True)
