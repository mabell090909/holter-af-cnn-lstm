import os
import time
import json
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Optimize threading
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(4)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
cache_file = os.path.join(RESULTS_DIR, 'ablation_cohort_dataset.npz')

WINDOW_SIZE = 1280

print("==========================================================================", flush=True)
print(" RIGOROUS ABLATION STUDY: PREPROCESSING DSP vs CLASS IMBALANCE (SMOTE)   ", flush=True)
print(" 3 Random Record-Level Shuffles (Mean \u00b1 SD, Zero Patient Leakage)     ", flush=True)
print("==========================================================================", flush=True)

# -----------------------------------------------------------------------------
# 1. LOAD PRE-EXTRACTED PAIRED ABLATION DATASET
# -----------------------------------------------------------------------------
print("\n[1/3] Loading cached 20-patient paired ablation dataset...", flush=True)
cohort_data = np.load(cache_file)
X_raw_all = cohort_data['X_raw']
X_bp_all = cohort_data['X_bp']
X_dsp_all = cohort_data['X_dsp']
y_dsp_all = cohort_data['y_dsp']
y_raw_all = cohort_data['y_raw']
subjects_dsp_all = cohort_data['subjects_dsp']
subjects_raw_all = cohort_data['subjects_raw']

unique_patients = np.unique(subjects_dsp_all)
print(f"   Total Windows : {len(X_dsp_all):,} across {len(unique_patients)} unique patients", flush=True)
print(f"   Overall Balance: {np.sum(y_dsp_all==1):,} AF ({np.mean(y_dsp_all)*100:.1f}%), {np.sum(y_dsp_all==0):,} Non-AF", flush=True)

# Categorize patients by AF presence for balanced stratified patient-level partitioning
pat_af_counts = {p: int(np.sum((subjects_dsp_all == p) & (y_dsp_all == 1))) for p in unique_patients}
pats_with_af = [p for p in unique_patients if pat_af_counts[p] > 50]
pats_without_af = [p for p in unique_patients if pat_af_counts[p] <= 50]

print(f"   Patient Rhythm Strata: {len(pats_with_af)} AF Patients, {len(pats_without_af)} Sinus/Low-AF Patients", flush=True)

# -----------------------------------------------------------------------------
# 2. MODEL ARCHITECTURE BUILDER
# -----------------------------------------------------------------------------
def build_ablation_model():
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

# -----------------------------------------------------------------------------
# 3. EXPERIMENTAL CONDITIONS
# -----------------------------------------------------------------------------
SEEDS = [42, 101, 2024]

CONDITIONS = [
    # Dimension 1: Preprocessing Filter Cascade
    {
        'id': 'C1_Raw_Unweighted',
        'name': 'Raw ECG (No DSP Filtering, Unweighted)',
        'filter': 'raw',
        'balance': 'unweighted'
    },
    {
        'id': 'C2_Bandpass_Unweighted',
        'name': 'Bandpass-Only (0.5-40 Hz, No QG, Unweighted)',
        'filter': 'bandpass',
        'balance': 'unweighted'
    },
    {
        'id': 'C3_FullDSP_Unweighted',
        'name': 'Full DSP Cascade (Dual-Stage + QG, Unweighted)',
        'filter': 'full_dsp',
        'balance': 'unweighted'
    },
    # Dimension 2: Class Imbalance Mitigation (evaluated on Full DSP Cascade)
    {
        'id': 'C4_FullDSP_SMOTE',
        'name': 'Full DSP Cascade + SMOTE Oversampling',
        'filter': 'full_dsp',
        'balance': 'smote'
    },
    {
        'id': 'C5_FullDSP_ClassWeight',
        'name': 'Full DSP Cascade + Cost-Sensitive Weight (Proposed)',
        'filter': 'full_dsp',
        'balance': 'class_weight'
    }
]

raw_runs = []
print(f"\n[2/3] Executing {len(SEEDS)} Random Patient-Level Shuffles x {len(CONDITIONS)} Conditions ({len(SEEDS)*len(CONDITIONS)} runs total)...", flush=True)

exp_counter = 0

for seed in SEEDS:
    print(f"\n=======================================================", flush=True)
    print(f" >>> RUNNING RANDOM PATIENT SHUFFLE: SEED {seed}", flush=True)
    print(f"=======================================================", flush=True)
    
    rng = np.random.RandomState(seed)
    shuffled_af = rng.permutation(pats_with_af)
    shuffled_sinus = rng.permutation(pats_without_af)
    
    # 70% Train (7 AF + 7 Sinus), 30% Test (3 AF + 3 Sinus)
    n_af_test = max(1, int(len(pats_with_af) * 0.30))
    n_sinus_test = max(1, int(len(pats_without_af) * 0.30))
    
    test_patients = list(shuffled_af[:n_af_test]) + list(shuffled_sinus[:n_sinus_test])
    train_patients = list(shuffled_af[n_af_test:]) + list(shuffled_sinus[n_sinus_test:])
    
    assert len(set(train_patients).intersection(set(test_patients))) == 0, "Patient leakage detected!"
    
    # Split train further into train_inner and val_inner for threshold tuning
    val_patients = [train_patients[0], train_patients[-1]]
    train_inner_patients = train_patients[1:-1]
    
    # Masks
    tr_mask_dsp = np.isin(subjects_dsp_all, train_inner_patients)
    val_mask_dsp = np.isin(subjects_dsp_all, val_patients)
    te_mask_dsp = np.isin(subjects_dsp_all, test_patients)
    
    tr_mask_raw = np.isin(subjects_raw_all, train_inner_patients)
    val_mask_raw = np.isin(subjects_raw_all, val_patients)
    te_mask_raw = np.isin(subjects_raw_all, test_patients)
    
    # Sample up to 200 windows per training patient for high throughput and stable gradients
    def sample_windows(mask, subjects_arr, max_w=200):
        idx = np.where(mask)[0]
        sel = []
        for p in np.unique(subjects_arr[idx]):
            p_idx = idx[subjects_arr[idx] == p]
            if len(p_idx) > max_w:
                sel.extend(rng.choice(p_idx, size=max_w, replace=False))
            else:
                sel.extend(p_idx)
        return np.array(sel, dtype=np.int64)
        
    tr_idx_dsp = sample_windows(tr_mask_dsp, subjects_dsp_all, 200)
    tr_idx_raw = sample_windows(tr_mask_raw, subjects_raw_all, 200)
    
    val_idx_dsp = np.where(val_mask_dsp)[0]
    te_idx_dsp = np.where(te_mask_dsp)[0]
    
    val_idx_raw = np.where(val_mask_raw)[0]
    te_idx_raw = np.where(te_mask_raw)[0]
    
    for cond in CONDITIONS:
        exp_counter += 1
        t_start = time.time()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        
        # 1. Select data based on filter condition
        if cond['filter'] == 'raw':
            X_tr = X_raw_all[tr_idx_raw]
            y_tr = y_raw_all[tr_idx_raw]
            X_val = X_raw_all[val_idx_raw]
            y_val = y_raw_all[val_idx_raw]
            X_te = X_raw_all[te_idx_raw]
            y_te = y_raw_all[te_idx_raw]
        elif cond['filter'] == 'bandpass':
            X_tr = X_bp_all[tr_idx_raw]
            y_tr = y_raw_all[tr_idx_raw]
            X_val = X_bp_all[val_idx_raw]
            y_val = y_raw_all[val_idx_raw]
            X_te = X_bp_all[te_idx_raw]
            y_te = y_raw_all[te_idx_raw]
        else: # full_dsp
            X_tr = X_dsp_all[tr_idx_dsp]
            y_tr = y_dsp_all[tr_idx_dsp]
            X_val = X_dsp_all[val_idx_dsp]
            y_val = y_dsp_all[val_idx_dsp]
            X_te = X_dsp_all[te_idx_dsp]
            y_te = y_dsp_all[te_idx_dsp]
            
        # 2. Balancing strategy
        cw_dict = None
        if cond['balance'] == 'class_weight':
            classes = np.unique(y_tr)
            weights = compute_class_weight('balanced', classes=classes, y=y_tr)
            cw_dict = {c: float(w) for c, w in zip(classes, weights)}
        elif cond['balance'] == 'smote':
            smote = SMOTE(random_state=seed, k_neighbors=3)
            n_samples = len(X_tr)
            X_2d = X_tr.reshape(n_samples, -1)
            X_res_2d, y_tr = smote.fit_resample(X_2d, y_tr)
            X_tr = X_res_2d.reshape(-1, WINDOW_SIZE, 1)
            
        # 3. Fast & robust 2-epoch training
        model = build_ablation_model()
        model.fit(
            X_tr, y_tr,
            batch_size=256,
            epochs=2,
            class_weight=cw_dict,
            verbose=0
        )
        
        # 4. Predict on val to select optimal operating threshold T*
        val_probs = model.predict(X_val, batch_size=512, verbose=0).ravel()
        cand_th = np.linspace(0.2, 0.85, 20)
        best_th, best_val_f1 = 0.5, -1.0
        for th in cand_th:
            p = (val_probs >= th).astype(int)
            f = f1_score(y_val, p, zero_division=0)
            if f > best_val_f1:
                best_val_f1 = f
                best_th = float(th)
                
        # 5. Predict on test
        test_probs = model.predict(X_te, batch_size=512, verbose=0).ravel()
        pred_labels = (test_probs >= best_th).astype(int)
        
        auc_val = float(roc_auc_score(y_te, test_probs)) if len(np.unique(y_te)) > 1 else 0.5
        acc_val = float(accuracy_score(y_te, pred_labels)) * 100.0
        f1_val = float(f1_score(y_te, pred_labels, zero_division=0)) * 100.0
        sens_val = float(recall_score(y_te, pred_labels, zero_division=0)) * 100.0
        spec_val = float(recall_score(1 - y_te, 1 - pred_labels, zero_division=0)) * 100.0
        prec_val = float(precision_score(y_te, pred_labels, zero_division=0)) * 100.0
        
        elapsed = time.time() - t_start
        
        raw_runs.append({
            'seed': seed,
            'condition_id': cond['id'],
            'condition_name': cond['name'],
            'filter_stage': cond['filter'],
            'balancing_strategy': cond['balance'],
            'optimal_threshold': round(best_th, 4),
            'roc_auc': round(auc_val, 4),
            'f1_score': round(f1_val, 2),
            'accuracy': round(acc_val, 2),
            'sensitivity': round(sens_val, 2),
            'specificity': round(spec_val, 2),
            'precision': round(prec_val, 2),
            'runtime_sec': round(elapsed, 1)
        })
        
        print(f"[{exp_counter:02d}/15] Seed {seed} | {cond['name']:<50} | "
              f"T*={best_th:.2f} | AUC: {auc_val:.4f} | F1: {f1_val:.2f}% | Acc: {acc_val:.2f}% | Sens: {sens_val:.2f}% | Spec: {spec_val:.2f}% ({elapsed:.1f}s)", flush=True)
        
        del model
        gc.collect()

# -----------------------------------------------------------------------------
# 4. STATISTICAL AGGREGATION & EXPORT
# -----------------------------------------------------------------------------
print("\n[3/3] Aggregating empirical statistics (Mean \u00b1 Sample SD, ddof=1)...", flush=True)

df_raw = pd.DataFrame(raw_runs)
raw_runs_csv = os.path.join(RESULTS_DIR, 'ablation_study_raw_runs.csv')
df_raw.to_csv(raw_runs_csv, index=False)
print(f"   -> Saved raw runs to {raw_runs_csv}", flush=True)

summary_rows = []
for cond in CONDITIONS:
    cid = cond['id']
    sub = df_raw[df_raw['condition_id'] == cid]
    
    mean_auc, std_auc = sub['roc_auc'].mean(), sub['roc_auc'].std(ddof=1)
    mean_f1, std_f1 = sub['f1_score'].mean(), sub['f1_score'].std(ddof=1)
    mean_acc, std_acc = sub['accuracy'].mean(), sub['accuracy'].std(ddof=1)
    mean_sens, std_sens = sub['sensitivity'].mean(), sub['sensitivity'].std(ddof=1)
    mean_spec, std_spec = sub['specificity'].mean(), sub['specificity'].std(ddof=1)
    mean_prec, std_prec = sub['precision'].mean(), sub['precision'].std(ddof=1)
    mean_th = sub['optimal_threshold'].mean()
    
    summary_rows.append({
        'condition_id': cid,
        'condition_name': cond['name'],
        'filter_stage': cond['filter'],
        'balancing_strategy': cond['balance'],
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
print(f"   -> Saved summary CSV to {summary_csv}", flush=True)

summary_json = os.path.join(RESULTS_DIR, 'ablation_study_summary.json')
with open(summary_json, 'w', encoding='utf-8') as f:
    json.dump({
        'study_title': 'Ablation Analysis: Preprocessing DSP Cascade vs Class Imbalance Handling (SMOTE)',
        'shuffles_count': len(SEEDS),
        'seeds_evaluated': SEEDS,
        'patient_leakage_guarantee': 'Zero leakage (Patient-Level Partitioning)',
        'threshold_calibration': 'Validation-tuned optimal operating point (T*)',
        'standard_deviation_convention': 'Unbiased sample standard deviation (ddof=1)',
        'summary_results': summary_rows
    }, f, indent=4)
print(f"   -> Saved summary JSON to {summary_json}", flush=True)

# Markdown Table
md_table = """### Table R6: Comprehensive Ablation Study: Preprocessing Stages & Imbalance Mitigation
*Empirical evaluation across 3 independent random patient-level shuffles with zero patient leakage. Results reported as Mean \u00b1 SD (ddof=1).*

| Preprocessing Stage | Balancing Strategy | Calibrated $T^*$ | ROC-AUC | F1-Score (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) | Precision (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
for r in summary_rows:
    md_table += f"| {r['condition_name'].split('(')[0].strip()} | {r['balancing_strategy'].replace('_', ' ').title()} | {r['mean_optimal_threshold']:.2f} | **{r['formatted_auc']}** | {r['formatted_f1']} | {r['formatted_acc']} | {r['formatted_sens']} | {r['formatted_spec']} | {r['formatted_prec']} |\n"

md_path = os.path.join(RESULTS_DIR, 'ablation_study_markdown_table.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_table)

print("\n" + md_table)
print("\n[SUCCESS] Ablation study completely finished and verified!", flush=True)
