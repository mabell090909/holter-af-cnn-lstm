import os
import glob
import json
import math
import shutil
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample_poly, find_peaks
from pathlib import Path
import concurrent.futures

# ==============================================================================
# DIREKTORI & PARAMETER STANDAR PIPELINE EKG KLINIS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw_data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
SUBJECTS_DIR = os.path.join(PROCESSED_DIR, 'subjects')

TARGET_FS = 128
WINDOW_SEC = 10
WINDOW_SIZE = TARGET_FS * WINDOW_SEC
EXCLUSION_FALLBACK_THRESHOLD = 20.0  # Batas eksklusi rekaman: Fallback >= 20.0%

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SUBJECTS_DIR, exist_ok=True)

def apply_bandpass_filter(data, fs=128, lowcut=0.5, highcut=40.0, order=3):
    """
    Filter Butterworth Bandpass (0.5 - 40.0 Hz) orde 3 zero-phase.
    Diterapkan pada sampling rate native sebelum resampling untuk mencegah aliasing.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def pan_tompkins_style_continuous(sig, fs=128):
    """
    Detektor QRS gaya Pan-Tompkins pada sinyal kontinu per rekaman:
    1. Filter bandpass 5-15 Hz (fokus energi spektral QRS)
    2. Five-point derivative & squaring
    3. Moving Window Integration (150 ms)
    4. Global threshold (mean + 0.5 * std dari sinyal terintegrasi)
    """
    nyq = 0.5 * fs
    b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype='band')
    sig_bp = filtfilt(b, a, np.nan_to_num(sig))
    
    diff_sig = np.diff(sig_bp)
    sq_sig = diff_sig ** 2
    
    win_size = max(1, int(0.15 * fs))
    kernel = np.ones(win_size) / win_size
    mwi = np.convolve(sq_sig, kernel, mode='same')
    
    threshold = np.mean(mwi) + 0.5 * np.std(mwi)
    min_dist = max(1, int(0.3 * fs))
    peaks, _ = find_peaks(mwi, distance=min_dist, height=threshold)
    return peaks

def compute_rr_features_window(w_peaks, fs=128):
    """
    Menghitung 5 fitur statistik RR-interval klasik per jendela:
    Mean RR, SDNN, RMSSD, pNN50, dan Shannon Entropy.
    Dijamin memiliki >= 3 puncak QRS (tidak pernah nol).
    """
    if len(w_peaks) < 3:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
    rrs = np.diff(w_peaks) / float(fs)
    mean_rr = float(np.mean(rrs))
    sdnn = float(np.std(rrs))
    
    diff_rr = np.diff(rrs)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0
    pnn50 = float(np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr)) if len(diff_rr) > 0 else 0.0
    
    counts, _ = np.histogram(rrs, bins=5, density=True)
    probs = counts / (np.sum(counts) + 1e-8)
    entropy = float(-np.sum([p * np.log(p + 1e-8) for p in probs if p > 0]))
    
    return np.array([mean_rr, sdnn, rmssd, pnn50, entropy], dtype=np.float32)

def evaluate_channel_quality(sig, fs):
    """
    Menghitung fallback rate (% jendela 10-s dengan < 3 QRS peaks) secara objektif
    tanpa melihat label anotasi ritme (label-blind).
    """
    peaks = pan_tompkins_style_continuous(sig, fs=fs)
    w_size = int(WINDOW_SEC * fs)
    n_win = len(sig) // w_size
    if n_win == 0:
        return 100.0, peaks
    bins = np.arange(n_win + 1) * w_size
    counts, _ = np.histogram(peaks, bins=bins)
    fb_count = int(np.sum(counts < 3))
    fb_pct = (fb_count / n_win) * 100.0
    return fb_pct, peaks

def find_record_dir(root):
    """
    Menemukan direktori yang berisi kumpulan file rekaman EKG terbanyak (.hea dan .dat).
    """
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

def get_available_records(db_path):
    """Mendeteksi rekaman lokal yang memiliki berkas lengkap (.hea, .dat, dan .atr)."""
    target_dir = find_record_dir(db_path)
    hea_files = glob.glob(os.path.join(target_dir, "*.hea"))
    valid_records = []
    for hea in hea_files:
        base_name = os.path.splitext(os.path.basename(hea))[0]
        dat_file = os.path.join(target_dir, f"{base_name}.dat")
        atr_file = os.path.join(target_dir, f"{base_name}.atr")
        if os.path.exists(dat_file) and os.path.exists(atr_file):
            if os.path.getsize(dat_file) > 0 and os.path.getsize(atr_file) > 0:
                valid_records.append(base_name)
    return target_dir, sorted(valid_records)

def process_record_worker(args):
    """
    Fungsi worker paralel untuk memproses 1 rekaman EKG:
    1. Evaluasi kualitas kanal 0 vs kanal 1 (label-blind).
    2. Pemilihan kanal dengan fallback terendah.
    3. Cek batas eksklusi rekaman: jika best_fb >= 20.0%, keluarkan rekaman.
    4. Terapkan Quality Gate Per Jendela: buang jendela dengan < 3 puncak QRS.
    5. Simpan segmen bersih ke disk per rekaman (zero memory accumulation).
    """
    rec_name, db_dir, db_name, patient_tag = args
    rec_tag = f"{db_name}_{rec_name}"
    rec_path = os.path.join(db_dir, rec_name)
    
    try:
        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, 'atr')
        
        fs = record.fs
        n_channels = record.n_sig if hasattr(record, 'n_sig') else record.p_signal.shape[1]
        
        sig0 = record.p_signal[:, 0]
        if np.any(np.isnan(sig0)) or np.any(np.isinf(sig0)):
            sig0 = np.nan_to_num(sig0, nan=0.0, posinf=0.0, neginf=0.0)
            
        fb0, peaks0 = evaluate_channel_quality(sig0, fs)
        
        if n_channels > 1:
            sig1 = record.p_signal[:, 1]
            if np.any(np.isnan(sig1)) or np.any(np.isinf(sig1)):
                sig1 = np.nan_to_num(sig1, nan=0.0, posinf=0.0, neginf=0.0)
            fb1, peaks1 = evaluate_channel_quality(sig1, fs)
        else:
            fb1 = 999.0
            sig1 = None
            peaks1 = None
            
        # Pemilihan kanal terbaik berdasarkan fallback terendah
        if fb0 <= fb1:
            best_ch = 0
            best_fb = fb0
            raw_chosen_sig = sig0
        else:
            best_ch = 1
            best_fb = fb1
            raw_chosen_sig = sig1
            
        lead_name = record.sig_name[best_ch] if getattr(record, 'sig_name', None) and len(record.sig_name) > best_ch else f'Ch_{best_ch}'
        
        # Hitung AF Burden dari anotasi ritme kardiolog
        aux_notes = getattr(ann, 'aux_note', None)
        ann_samples_raw = ann.sample
        total_rec_len = len(raw_chosen_sig)
        
        labels_raw = np.zeros(total_rec_len, dtype=np.int32)
        curr_lbl = 0
        for i, s_idx in enumerate(ann_samples_raw):
            note = aux_notes[i] if aux_notes is not None and i < len(aux_notes) else ''
            clean = note.replace('\x00', '').strip().upper() if isinstance(note, str) else ''
            if clean.startswith('('):
                curr_lbl = 1 if clean.startswith(('(AFIB', '(AFL')) else 0
            next_idx = ann_samples_raw[i + 1] if i + 1 < len(ann_samples_raw) else total_rec_len
            st = max(0, min(s_idx, total_rec_len))
            en = max(0, min(next_idx, total_rec_len))
            if st < en:
                labels_raw[st:en] = curr_lbl
                
        rec_af_samples = int(np.sum(labels_raw == 1))
        rec_af_burden_pct = round((rec_af_samples / max(1, total_rec_len)) * 100.0, 2)
        
        # CEK ATURAN EKSKLUSI REKAMAN: Fallback >= 20.0%
        if best_fb >= EXCLUSION_FALLBACK_THRESHOLD:
            return {
                'status': 'EXCLUDED',
                'database': db_name,
                'record': rec_name,
                'rec_tag': rec_tag,
                'patient_tag': patient_tag,
                'best_channel': best_ch,
                'ch0_fallback_pct': round(fb0, 2),
                'ch1_fallback_pct': round(fb1, 2) if n_channels > 1 else None,
                'best_fallback_pct': round(best_fb, 2),
                'af_burden_pct': rec_af_burden_pct,
                'lead_name': lead_name,
                'reason': f"Best channel fallback ({best_fb:.2f}%) >= {EXCLUSION_FALLBACK_THRESHOLD}%"
            }
            
        # REKAMAN LOLOS -> PROSES PENYARINGAN & QUALITY GATE JENDELA
        # 1. Bandpass filter native
        sig_filtered = apply_bandpass_filter(raw_chosen_sig, fs=fs)
        
        # 2. Resampling polyphase ke TARGET_FS (128 Hz)
        if int(round(fs)) != TARGET_FS:
            g = math.gcd(TARGET_FS, int(round(fs)))
            up = TARGET_FS // g
            down = int(round(fs)) // g
            sig_resampled = resample_poly(sig_filtered, up, down)
            ann_samples = np.round(ann.sample * up / down).astype(int)
        else:
            sig_resampled = sig_filtered
            ann_samples = ann.sample
            
        # 3. Pan-Tompkins kontinu pada sinyal kontinu 128 Hz
        continuous_qrs_peaks = pan_tompkins_style_continuous(sig_resampled, fs=TARGET_FS)
        
        # 4. Labeling ritme pada sinyal 128 Hz
        labels = np.zeros(len(sig_resampled), dtype=np.int32)
        curr_lbl = 0
        for i, s_idx in enumerate(ann_samples):
            note = aux_notes[i] if aux_notes is not None and i < len(aux_notes) else ''
            clean = note.replace('\x00', '').strip().upper() if isinstance(note, str) else ''
            if clean.startswith('('):
                curr_lbl = 1 if clean.startswith(('(AFIB', '(AFL')) else 0
            next_idx = ann_samples[i + 1] if i + 1 < len(ann_samples) else len(sig_resampled)
            st = max(0, min(s_idx, len(labels)))
            en = max(0, min(next_idx, len(labels)))
            if st < en:
                labels[st:en] = curr_lbl
                
        # 5. Segmentasi Jendela 10-Detik (1280 Sampel) & QUALITY GATE PER JENDELA (<3 Puncak QRS Ditolak)
        n_total_windows = len(sig_resampled) // WINDOW_SIZE
        retained_x = []
        retained_rr = []
        retained_y = []
        unreadable_windows_count = 0
        
        for w_idx in range(n_total_windows):
            start = w_idx * WINDOW_SIZE
            end = start + WINDOW_SIZE
            
            # Cari puncak QRS pada jendela ini
            w_peaks = continuous_qrs_peaks[(continuous_qrs_peaks >= start) & (continuous_qrs_peaks < end)] - start
            
            # QUALITY GATE PER JENDELA: Jendela dengan < 3 puncak QRS adalah sinyal tidak terbaca / artefak bising
            if len(w_peaks) < 3:
                unreadable_windows_count += 1
                continue  # Dikeluarkan dari dataset latih & uji
                
            window_sig = sig_resampled[start:end]
            window_labels = labels[start:end]
            
            std_val = np.std(window_sig)
            if std_val < 1e-6:
                norm_sig = window_sig - np.mean(window_sig)
            else:
                norm_sig = (window_sig - np.mean(window_sig)) / std_val
                
            is_afib = 1 if np.mean(window_labels) >= 0.5 else 0
            rr_feat = compute_rr_features_window(w_peaks, fs=TARGET_FS)
            
            retained_x.append(norm_sig)
            retained_rr.append(rr_feat)
            retained_y.append(is_afib)
            
        n_retained = len(retained_x)
        if n_retained > 0:
            X_arr = np.array(retained_x, dtype=np.float32)[:, :, np.newaxis]
            rr_arr = np.array(retained_rr, dtype=np.float32)
            y_arr = np.array(retained_y, dtype=np.int32)
            
            sub_file = os.path.join(SUBJECTS_DIR, f"{rec_tag}.npz")
            np.savez_compressed(sub_file, X=X_arr, rr_features=rr_arr, y=y_arr)
            
            n_non_af = int(np.sum(y_arr == 0))
            n_af = int(np.sum(y_arr == 1))
        else:
            n_non_af = 0
            n_af = 0
            
        return {
            'status': 'RETAINED',
            'database': db_name,
            'record': rec_name,
            'rec_tag': rec_tag,
            'patient_tag': patient_tag,
            'original_sampling_rate_hz': fs,
            'best_channel': best_ch,
            'lead_channel_used': lead_name,
            'ch0_fallback_pct': round(fb0, 2),
            'ch1_fallback_pct': round(fb1, 2) if n_channels > 1 else None,
            'best_fallback_pct': round(best_fb, 2),
            'continuous_qrs_peaks_detected': len(continuous_qrs_peaks),
            'total_windows_10s_extracted': n_total_windows,
            'unreadable_windows_rejected': unreadable_windows_count,
            'unreadable_pct': round((unreadable_windows_count / max(1, n_total_windows)) * 100.0, 2),
            'retained_windows': n_retained,
            'non_af_count': n_non_af,
            'afib_count': n_af,
            'af_burden_pct': round((n_af / max(1, n_retained)) * 100.0, 2)
        }
        
    except Exception as e:
        return {
            'status': 'FAILED',
            'database': db_name,
            'record': rec_name,
            'rec_tag': rec_tag,
            'patient_tag': patient_tag,
            'reason': str(e)
        }

def run_preprocessing():
    print("========================================================================")
    print(" MEMULAI PIPELINE PREPROCESSING EKG KLINIS MUTAKHIR (ZERO-LEAKAGE + SQI)")
    print(" 1. Pemilihan Kanal Terbaik: Label-Blind Quality Selection (Lowest Fallback)")
    print(" 2. Eksklusi Rekaman Rusak: Ambang Batas Fallback >= 20.0% pada Kanal Terbaik")
    print(" 3. Quality Gate Per Jendela: Menolak Jendela dengan < 3 Puncak QRS")
    print(" 4. Bebas RAM OOM: Zero-Memory Subject Saving + Structured Memmap Streaming")
    print("========================================================================")
    
    datasets = {
        'ltafdb': os.path.join(RAW_DIR, 'ltafdb'),
        'shdb-af': os.path.join(RAW_DIR, 'shdb-af')
    }
    
    # Bersihkan subjects/ NPZ lama agar tidak tercampur
    for f in glob.glob(os.path.join(SUBJECTS_DIR, "*.npz")):
        try:
            os.remove(f)
        except OSError:
            pass
            
    # Muat metadata pemetaan pasien SHDB-AF dari AdditionalData.csv
    shdb_patient_map = {}
    csv_candidates = list(Path(RAW_DIR).rglob('AdditionalData.csv'))
    if csv_candidates:
        csv_candidates.sort(key=lambda p: len(p.parts), reverse=True)
        chosen_csv = csv_candidates[0]
        df_meta = pd.read_csv(chosen_csv)
        for _, row in df_meta.iterrows():
            try:
                d_id = int(row['Data_ID'])
                s_id = str(row['Subject_ID']).strip()
                shdb_patient_map[d_id] = s_id
            except Exception:
                pass
        print(f"[METADATA] Memuat pemetaan pasien SHDB-AF dari: {chosen_csv} ({len(shdb_patient_map)} Data_ID terdaftar).", flush=True)
        if len(csv_candidates) > 1:
            df_other = pd.read_csv(csv_candidates[-1])
            assert len(df_meta) == len(df_other), "Peringatan: Jumlah baris AdditionalData.csv berbeda!"
            print(f"  [VERIFIKASI METADATA] Berkas AdditionalData.csv konsisten ({len(df_meta)} baris).", flush=True)
            
    # Kumpulkan seluruh daftar tugas rekaman
    tasks = []
    for db_name, db_path in datasets.items():
        if not os.path.exists(db_path):
            continue
        target_dir, records = get_available_records(db_path)
        print(f"Basis Data [{db_name}]: Ditemukan {len(records)} rekaman lengkap di {target_dir}")
        for rec in records:
            if db_name == 'shdb-af':
                assert len(shdb_patient_map) > 0, "AdditionalData.csv tidak ditemukan!"
                rec_int = int(rec)
                assert rec_int in shdb_patient_map, f"Rekaman {rec} tidak ada di AdditionalData.csv"
                p_tag = f"shdb-af_patient_{shdb_patient_map[rec_int]}"
            else:
                p_tag = f"ltafdb_patient_{rec}"
            tasks.append((rec, target_dir, db_name, p_tag))
            
    print(f"\nMenjalankan preprocessing paralel untuk {len(tasks)} rekaman EKG...")
    
    # Jalankan worker secara paralel menggunakan multi-processing
    workers = min(16, max(1, os.cpu_count() - 2 if os.cpu_count() else 4))
    print(f"Mengalokasikan {workers} worker proses simultan...")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_record_worker, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            status = res['status']
            db = res['database']
            rec = res['record']
            if status == 'RETAINED':
                print(f"  [{db}] {rec} -> Ch {res['best_channel']} (FB: {res['best_fallback_pct']}%) | Jendela: {res['retained_windows']:,} (Ditolak: {res['unreadable_windows_rejected']:,})", flush=True)
            elif status == 'EXCLUDED':
                print(f"  [EKSKLUSI] [{db}] {rec} -> DIKELUARKAN: {res['reason']} (AF Burden: {res['af_burden_pct']}%)", flush=True)
            else:
                print(f"  [GAGAL] [{db}] {rec} -> Gagal: {res.get('reason')}", flush=True)
                
    # Pisahkan hasil
    retained_list = [r for r in results if r['status'] == 'RETAINED']
    excluded_list = [r for r in results if r['status'] == 'EXCLUDED']
    failed_list = [r for r in results if r['status'] == 'FAILED']
    
    # Sort retained records secara deterministik berdasarkan rec_tag
    retained_list.sort(key=lambda x: x['rec_tag'])
    excluded_list.sort(key=lambda x: x['rec_tag'])
    
    total_retained_samples = sum(r['retained_windows'] for r in retained_list)
    total_non_af = sum(r['non_af_count'] for r in retained_list)
    total_afib = sum(r['afib_count'] for r in retained_list)
    
    print(f"\n========================================================================")
    print(f" SELEKSI REKAMAN & QUALITY GATE JENDELA SELESAI")
    print(f" Total Rekaman Dianalisis: {len(results)}")
    print(f" Rekaman Dipertahankan   : {len(retained_list)}")
    print(f" Rekaman Dikeluarkan     : {len(excluded_list)}")
    print(f" Rekaman Gagal           : {len(failed_list)}")
    print(f" Total Jendela Bersih    : {total_retained_samples:,}")
    print(f"========================================================================")
    
    # PENYUSUNAN MEMMAP TENSOR
    print(f"\n[PENYUSUNAN MEMMAP] Menulis {total_retained_samples:,} jendela bersih ke berkas memmap...")
    x_memmap_path = os.path.join(PROCESSED_DIR, 'X.npy')
    rr_memmap_path = os.path.join(PROCESSED_DIR, 'rr_features.npy')
    y_memmap_path = os.path.join(PROCESSED_DIR, 'y.npy')
    sub_memmap_path = os.path.join(PROCESSED_DIR, 'subjects.npy')
    
    # Hapus file lama jika ada
    for p in [x_memmap_path, rr_memmap_path, y_memmap_path, sub_memmap_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
                
    X_mmap = np.lib.format.open_memmap(x_memmap_path, mode='w+', dtype=np.float32, shape=(total_retained_samples, WINDOW_SIZE, 1))
    rr_mmap = np.lib.format.open_memmap(rr_memmap_path, mode='w+', dtype=np.float32, shape=(total_retained_samples, 5))
    y_mmap = np.lib.format.open_memmap(y_memmap_path, mode='w+', dtype=np.int32, shape=(total_retained_samples,))
    subjects_all = []
    
    curr_offset = 0
    for r in retained_list:
        sub_npz = os.path.join(SUBJECTS_DIR, f"{r['rec_tag']}.npz")
        n_w = r['retained_windows']
        with np.load(sub_npz) as data:
            X_mmap[curr_offset : curr_offset + n_w] = data['X']
            rr_mmap[curr_offset : curr_offset + n_w] = data['rr_features']
            y_mmap[curr_offset : curr_offset + n_w] = data['y']
            subjects_all.extend([r['patient_tag']] * n_w)
        curr_offset += n_w
        
    X_mmap.flush()
    rr_mmap.flush()
    y_mmap.flush()
    np.save(sub_memmap_path, np.array(subjects_all, dtype=object))
    
    # Perhitungan Statistik Basis Data
    db_summary_stats = {}
    channel_selection_summary = {}
    
    for db_name in ['ltafdb', 'shdb-af']:
        db_ret = [r for r in retained_list if r['database'] == db_name]
        db_exc = [r for r in excluded_list if r['database'] == db_name]
        db_patients = set(r['patient_tag'] for r in db_ret)
        
        tot_ext = sum(r['total_windows_10s_extracted'] for r in db_ret)
        tot_unred = sum(r['unreadable_windows_rejected'] for r in db_ret)
        tot_ret = sum(r['retained_windows'] for r in db_ret)
        af_cnt = sum(r['afib_count'] for r in db_ret)
        non_af_cnt = sum(r['non_af_count'] for r in db_ret)
        
        ch0_cnt = sum(1 for r in db_ret if r['best_channel'] == 0)
        ch1_cnt = sum(1 for r in db_ret if r['best_channel'] == 1)
        
        db_summary_stats[db_name] = {
            'records_evaluated': len(db_ret) + len(db_exc),
            'records_retained': len(db_ret),
            'records_excluded': len(db_exc),
            'unique_patients_retained': len(db_patients),
            'channel_0_chosen': ch0_cnt,
            'channel_1_chosen': ch1_cnt,
            'total_windows_extracted': tot_ext,
            'unreadable_windows_rejected': tot_unred,
            'unreadable_rejection_pct': round((tot_unred / max(1, tot_ext)) * 100.0, 2),
            'retained_valid_windows': tot_ret,
            'af_windows': af_cnt,
            'non_af_windows': non_af_cnt,
            'af_pct': round((af_cnt / max(1, tot_ret)) * 100.0, 2)
        }
        channel_selection_summary[db_name] = {
            'channel_0': ch0_cnt,
            'channel_1': ch1_cnt
        }
        
    # Analisis AF Burden Tingkat Pasien
    unique_patients_all = sorted(list(set(subjects_all)))
    patient_burdens = {}
    for p in unique_patients_all:
        p_recs = [r for r in retained_list if r['patient_tag'] == p]
        p_total_win = sum(r['retained_windows'] for r in p_recs)
        p_af_win = sum(r['afib_count'] for r in p_recs)
        burden = round((p_af_win / max(1, p_total_win)) * 100.0, 2)
        patient_burdens[p] = {
            'records': [r['record'] for r in p_recs],
            'total_windows': p_total_win,
            'af_windows': p_af_win,
            'burden_pct': burden
        }
        
    burden_values = [v['burden_pct'] for v in patient_burdens.values()]
    burden_zero_count = sum(1 for b in burden_values if b == 0.0)
    burden_hundred_count = sum(1 for b in burden_values if b == 100.0)
    multi_rec_patients = {p: v['records'] for p, v in patient_burdens.items() if len(v['records']) > 1}
    
    total_windows_extracted_all = sum(r['total_windows_10s_extracted'] for r in retained_list)
    total_unreadable_all = sum(r['unreadable_windows_rejected'] for r in retained_list)
    
    print("\n========================================================================")
    print(" RINGKASAN QUALITY GATE PREPROCESSING DATASET BERSIH (QUALITY-FILTERED)")
    print("========================================================================")
    for db_k, db_s in db_summary_stats.items():
        print(f" Basis Data: [{db_k.upper()}]")
        print(f"   - Rekaman Dianalisis      : {db_s['records_evaluated']}")
        print(f"   - Rekaman Dipertahankan   : {db_s['records_retained']}")
        print(f"   - Rekaman Dikeluarkan     : {db_s['records_excluded']} (Fallback >= {EXCLUSION_FALLBACK_THRESHOLD}%)")
        print(f"   - Pasien Unik Dipertahankan: {db_s['unique_patients_retained']}")
        print(f"   - Pemilihan Kanal         : Kanal 0: {db_s['channel_0_chosen']}, Kanal 1: {db_s['channel_1_chosen']}")
        print(f"   - Total Jendela Mentah    : {db_s['total_windows_extracted']:,}")
        print(f"   - Jendela Tidak Terbaca   : {db_s['unreadable_windows_rejected']:,} ({db_s['unreadable_rejection_pct']}%) [DITOLAK]")
        print(f"   - Jendela Valid Bersih    : {db_s['retained_valid_windows']:,}")
        print(f"   - Jendela AF              : {db_s['af_windows']:,} ({db_s['af_pct']}%)")
        print(f"   - Jendela Non-AF          : {db_s['non_af_windows']:,} ({100 - db_s['af_pct']:.2f}%)")
        
    print(f"\n Agregat Seluruh Basis Data:")
    print(f"   - Total Rekaman Dipertahankan: {len(retained_list)} rekaman (dari {len(results)} rekaman)")
    print(f"   - Total Pasien Unik          : {len(unique_patients_all)} pasien")
    print(f"   - Pasien Multi-Rekaman       : {len(multi_rec_patients)} pasien ({multi_rec_patients})")
    print(f"   - Total Jendela Mentah       : {total_windows_extracted_all:,}")
    print(f"   - Jendela Tidak Terbaca (<3p): {total_unreadable_all:,} ({(total_unreadable_all/total_windows_extracted_all)*100:.2f}%) [DITOLAK]")
    print(f"   - Total Jendela Valid Bersih : {total_retained_samples:,} (100% Valid QRS)")
    print(f"   - Segmen AF (Kelas 1)        : {total_afib:,} ({(total_afib/total_retained_samples)*100:.2f}%)")
    print(f"   - Segmen Non-AF (Kelas 0)    : {total_non_af:,} ({(total_non_af/total_retained_samples)*100:.2f}%)")
    print(f"   - RR Interval Fallback Rate  : 0.00% (Semua jendela memiliki >= 3 QRS)")
    print(f"\n Profil Distribusi AF Burden Pasien:")
    print(f"   - Rentang AF Burden          : {min(burden_values):.2f}% s.d. {max(burden_values):.2f}%")
    print(f"   - Median AF Burden           : {float(np.median(burden_values)):.2f}%")
    print(f"   - Pasien Burden = 0.0%       : {burden_zero_count} pasien")
    print(f"   - Pasien Burden = 100.0%     : {burden_hundred_count} pasien")
    print(f"\n Daftar Rekaman yang Dikeluarkan ({len(excluded_list)}):")
    for ex in excluded_list:
        print(f"   * [{ex['database']}] {ex['record']} (Pasien: {ex['patient_tag']}) | Kanal Terbaik: {ex['best_channel']} | Fallback: {ex['best_fallback_pct']}% | AF Burden: {ex['af_burden_pct']}%")
    print("========================================================================")
    
    summary_data = {
        'quality_gate_passed': True,
        'exclusion_fallback_threshold_pct': EXCLUSION_FALLBACK_THRESHOLD,
        'total_evaluated_records': len(results),
        'retained_records_count': len(retained_list),
        'excluded_records_count': len(excluded_list),
        'excluded_records': excluded_list,
        'channel_selection_summary': channel_selection_summary,
        'total_windows_extracted': total_windows_extracted_all,
        'unreadable_windows_rejected': total_unreadable_all,
        'unreadable_rejection_pct': round((total_unreadable_all / max(1, total_windows_extracted_all)) * 100.0, 2),
        'retained_clean_samples': total_retained_samples,
        'unique_patients_count': len(unique_patients_all),
        'multi_record_patients': multi_rec_patients,
        'databases': db_summary_stats,
        'patient_burden_distribution': {
            'min_burden_pct': min(burden_values),
            'max_burden_pct': max(burden_values),
            'median_burden_pct': float(np.median(burden_values)),
            'patients_zero_burden': burden_zero_count,
            'patients_hundred_burden': burden_hundred_count
        },
        'failed_records': failed_list,
        'patient_burdens': patient_burdens,
        'record_details': {r['rec_tag']: r for r in retained_list}
    }
    
    with open(os.path.join(PROCESSED_DIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"\nBerkas tensor X.npy, rr_features.npy, y.npy, subjects.npy, dan summary.json sukses diperbarui di: {PROCESSED_DIR}/")

if __name__ == '__main__':
    run_preprocessing()
