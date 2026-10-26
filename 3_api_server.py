import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import sqlite3
import threading
import urllib.request
import json
import os
from scipy.signal import butter, filtfilt, resample, resample_poly, find_peaks

app = Flask(__name__)
CORS(app) # Mengizinkan Node.js mengakses API ini

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(BASE_DIR, 'cnn_lstm_model.keras')
if not os.path.exists(MODEL_NAME):
    MODEL_NAME = os.path.join(BASE_DIR, 'cnn_lstm_model.h5')
if not os.path.exists(MODEL_NAME):
    MODEL_NAME = os.path.join(os.path.dirname(BASE_DIR), 'cnn_lstm_model.keras')
if not os.path.exists(MODEL_NAME):
    MODEL_NAME = os.path.join(os.path.dirname(BASE_DIR), 'cnn_lstm_model.h5')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(1280, 1)):
    m = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
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
    return m

try:
    print("Memuat Model CNN-LSTM...")
    try:
        model = load_model(MODEL_NAME)
    except Exception as e:
        print(f"Menggunakan arsitektur model + load_weights karena format versi Keras ({e})...")
        model = build_model((1280, 1))
        model.load_weights(MODEL_NAME)
    print("Model berhasil dimuat! Siap menerima sinyal EKG.")
except Exception as e:
    print(f"Peringatan: Gagal memuat {MODEL_NAME}: {e}")
    model = None

FS = 128
WINDOW_SEC = 10
WINDOW_SIZE = FS * WINDOW_SEC

def apply_bandpass_filter(data, fs=128, lowcut=0.5, highcut=40.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    mean = np.mean(filtered)
    std = np.std(filtered)
    return (filtered - mean) / (std + 1e-8)

def count_qrs_peaks(sig, fs=128):
    """
    Menghitung jumlah puncak QRS gaya Pan-Tompkins pada jendela 10-detik.
    Quality Gate Klinis: Jendela dengan < 3 puncak QRS ditandai sebagai artefak/unreadable.
    """
    try:
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
        return len(peaks)
    except Exception:
        return 0

@app.route('/predict', methods=['POST'])
def predict_afib():
    if model is None:
        return jsonify({'error': 'Model belum dilatih!'}), 500
        
    data = request.json
    if not data or 'ecg_signal' not in data:
        return jsonify({'error': 'Parameter ecg_signal tidak ditemukan'}), 400
        
    signal = np.array(data['ecg_signal'])
    
    # --- RESAMPLING & FILTERING PIPELINE (Konsisten dengan 1_preprocess.py) ---
    INPUT_FS = data.get('sampling_rate', 250) 
    TARGET_FS = 128 # Target sampling rate model AI (128 Hz)
    
    # 1. Bersihkan sinyal pada native sampling rate (Anti-aliasing bandpass 0.5 - 40 Hz)
    signal_filtered = apply_bandpass_filter(signal, fs=INPUT_FS)
    
    # 2. Resampling ke TARGET_FS (128 Hz) menggunakan polyphase filter
    if int(round(INPUT_FS)) != TARGET_FS:
        import math
        g = math.gcd(TARGET_FS, int(round(INPUT_FS)))
        up = TARGET_FS // g
        down = int(round(INPUT_FS)) // g
        signal_processed = resample_poly(signal_filtered, up, down)
    else:
        signal_processed = signal_filtered
    
    # 3. Potong-potong sinyal per 10 detik (1280 sampel pada 128 Hz)
    raw_segments = []
    for i in range(0, len(signal_processed) - WINDOW_SIZE + 1, WINDOW_SIZE):
        w_sig = signal_processed[i : i + WINDOW_SIZE]
        raw_segments.append(w_sig)
        
    if len(raw_segments) == 0:
        return jsonify({'error': 'Sinyal terlalu pendek (minimal 10 detik)'}), 400
        
    # --- QUALITY GATE PER WINDOW (< 3 QRS Peaks Rejected as Noise/Lead-Off) ---
    valid_indices = []
    valid_segments = []
    unreadable_indices = []
    
    for idx, w_sig in enumerate(raw_segments):
        n_peaks = count_qrs_peaks(w_sig, fs=TARGET_FS)
        if n_peaks >= 3:
            std_val = np.std(w_sig)
            norm_sig = (w_sig - np.mean(w_sig)) / std_val if std_val >= 1e-6 else (w_sig - np.mean(w_sig))
            valid_segments.append(norm_sig)
            valid_indices.append(idx)
        else:
            unreadable_indices.append(idx)
            
    # Baca threshold terkalibrasi dari final_model_metadata.json
    calibrated_threshold = 0.5763
    meta_file = os.path.join(BASE_DIR, 'results', 'final_model_metadata.json')
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                calibrated_threshold = float(json.load(f).get('median_optimal_threshold', 0.5763))
        except Exception:
            pass
            
    AF_THRESHOLD = calibrated_threshold
    
    # Jika seluruh jendela tidak terbaca
    if len(valid_segments) == 0:
        return jsonify({
            'status': 'REJECTED_UNREADABLE',
            'error': 'Kualitas sinyal tidak terbaca (kurang dari 3 puncak QRS di setiap jendela 10-detik). Elektroda mungkin terlepas.',
            'total_evaluated_windows': len(raw_segments),
            'readable_windows': 0,
            'rejected_unreadable_windows': len(unreadable_indices)
        }), 422
        
    X = np.array(valid_segments, dtype=np.float32)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # 4. Masukkan ke Mesin AI
    predictions = model.predict(X)
    
    # 5. Susun Hasil Laporan Klinis & Pengelompokan Episode (>= 30 detik / >= 3 jendela berturut-turut)
    episodes = []
    current_run = []
    total_af_windows = 0
    window_results = []
    
    val_ptr = 0
    for idx in range(len(raw_segments)):
        if idx in valid_indices:
            prob = float(predictions[val_ptr][0])
            is_pos = (prob >= AF_THRESHOLD)
            val_ptr += 1
            window_results.append({
                'window_index': idx,
                'start_sec': idx * WINDOW_SEC,
                'end_sec': (idx + 1) * WINDOW_SEC,
                'status': 'READABLE',
                'predicted_class': 'AF' if is_pos else 'NON_AF',
                'af_probability': round(prob, 4)
            })
            if is_pos:
                total_af_windows += 1
                current_run.append((idx, prob))
            else:
                if len(current_run) >= 3:
                    start_w = current_run[0][0]
                    end_w = current_run[-1][0]
                    episodes.append({
                        'episode_id': len(episodes) + 1,
                        'start_sec': start_w * WINDOW_SEC,
                        'end_sec': (end_w + 1) * WINDOW_SEC,
                        'duration_sec': len(current_run) * WINDOW_SEC,
                        'windows_count': len(current_run),
                        'mean_model_score': round(float(np.mean([p for _, p in current_run])), 4)
                    })
                current_run = []
        else:
            window_results.append({
                'window_index': idx,
                'start_sec': idx * WINDOW_SEC,
                'end_sec': (idx + 1) * WINDOW_SEC,
                'status': 'UNREADABLE_ARTIFACT',
                'predicted_class': 'EXCLUDED',
                'af_probability': None
            })
            if len(current_run) >= 3:
                start_w = current_run[0][0]
                end_w = current_run[-1][0]
                episodes.append({
                    'episode_id': len(episodes) + 1,
                    'start_sec': start_w * WINDOW_SEC,
                    'end_sec': (end_w + 1) * WINDOW_SEC,
                    'duration_sec': len(current_run) * WINDOW_SEC,
                    'windows_count': len(current_run),
                    'mean_model_score': round(float(np.mean([p for _, p in current_run])), 4)
                })
            current_run = []
            
    if len(current_run) >= 3:
        start_w = current_run[0][0]
        end_w = current_run[-1][0]
        episodes.append({
            'episode_id': len(episodes) + 1,
            'start_sec': start_w * WINDOW_SEC,
            'end_sec': (end_w + 1) * WINDOW_SEC,
            'duration_sec': len(current_run) * WINDOW_SEC,
            'windows_count': len(current_run),
            'mean_model_score': round(float(np.mean([p for _, p in current_run])), 4)
        })
        
    total_analyzed_sec = len(valid_segments) * WINDOW_SEC
    af_episode_duration_sec = sum(ep['duration_sec'] for ep in episodes)
    has_clinical_af = len(episodes) > 0
    overall_status = "Episode Atrial Fibrillation Terdeteksi (durasi >= 30 detik)" if has_clinical_af else "AF Tidak Terdeteksi"
    
    result = {
        'total_analyzed_sec': total_analyzed_sec,
        'af_duration_sec': af_episode_duration_sec,
        'af_burden_percentage': round((af_episode_duration_sec / max(1, total_analyzed_sec)) * 100, 2),
        'episodes_detected': len(episodes),
        'is_af_detected': has_clinical_af,
        'overall_status': overall_status,
        'applied_threshold': round(AF_THRESHOLD, 4),
        'total_evaluated_windows': len(raw_segments),
        'readable_windows': len(valid_segments),
        'rejected_unreadable_windows': len(unreadable_indices),
        'isolated_af_windows_below_30s': max(0, total_af_windows - sum(ep['windows_count'] for ep in episodes)),
        'episodes_detail': episodes,
        'window_results': window_results,
        'clinical_disclaimer': 'Prototipe riset akademis terkalibrasi 5-fold CV (Zero Patient Leakage), bukan alat diagnosis medis resmi.'
    }
    
    return jsonify(result)

def background_task(session_id):
    try:
        print(f"Background task started for session: {session_id}")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.abspath(os.path.join(BASE_DIR, '../webserver/ekg_data.db'))
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT duration_minutes FROM sessions WHERE session_id = ?", (session_id,))
        sess_row = cursor.fetchone()
        duration_minutes = sess_row[0] if sess_row and sess_row[0] else None
        
        cursor.execute("SELECT ecg_waveform FROM ekg_data WHERE session_id = ? ORDER BY id ASC", (session_id,))
        
        signal_filtered_all = []
        batch_size = 5000
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                try:
                    wf = json.loads(row[0])
                    signal_filtered_all.extend(wf)
                except:
                    pass
        conn.close()

        if len(signal_filtered_all) == 0:
            print("No data found.")
            try:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:5001/api/sessions/{session_id}/ai-webhook", data=json.dumps({"error": "Tidak ada data EKG sama sekali untuk sesi ini. Pastikan Anda sudah merekam data menggunakan HP."}).encode('utf-8'), headers={'Content-Type': 'application/json'}))
            except:
                pass
            return

        signal = np.array(signal_filtered_all)
        print(f"Total signal points: {len(signal)}")

        # --- PEMBATASAN DURASI MAKSIMAL SESUAI SESI (Mencegah durasi tembus ke menit berikutnya) ---
        INPUT_FS = 250  # Kecepatan asli alat EKG (250 Hz)
        if duration_minutes and duration_minutes > 0:
            max_allowed_samples = int(duration_minutes * 60 * INPUT_FS)
            if len(signal) > max_allowed_samples:
                print(f"Memotong kelebihan data: {len(signal)} titik dipotong menjadi {max_allowed_samples} titik ({duration_minutes} menit tepat)")
                signal = signal[:max_allowed_samples]
        TARGET_FS = 128 # Kecepatan yang diharapkan model AI (128 Hz)
        
        duration_sec = len(signal) / INPUT_FS
        target_samples = int(duration_sec * TARGET_FS)
        print(f"Resampling sinyal dari {len(signal)} titik (250Hz) menjadi {target_samples} titik (128Hz)...")
        
        signal = resample_poly(signal, TARGET_FS, INPUT_FS)

        # Bersihkan sinyal setelah resample
        signal_filtered = apply_bandpass_filter(signal, TARGET_FS)
        
        # --- Signal Quality Assessment (SQA) Lapis 1: Flatline ---
        if np.std(signal_filtered) < 0.001:
            print("SQA Failed: Flatline detected.")
            try:
                urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:5001/api/sessions/{session_id}/ai-webhook", 
                    data=json.dumps({"error": "Sinyal Terlalu Datar (Flatline). Kemungkinan elektroda terlepas dari kulit atau alat mati. Analisis AI dibatalkan."}).encode('utf-8'), 
                    headers={'Content-Type': 'application/json'}
                ))
            except:
                pass
            return
        
        # 1. Gunakan jarak minimum 0.24 detik dan threshold terkalibrasi untuk membedakan R-peak dari gelombang T
        pos_signal = signal_filtered[signal_filtered > 0]
        r_height_thresh = max(0.45 * np.percentile(pos_signal, 98), np.mean(signal_filtered) + 1.2 * np.std(signal_filtered)) if len(pos_signal) > 0 else (np.mean(signal_filtered) + np.std(signal_filtered))
        peaks, _ = find_peaks(signal_filtered, distance=int(TARGET_FS * 0.24), height=r_height_thresh)
        
        if len(peaks) > 2 and duration_sec > 0:
            # 2. Rata-rata HR (ekuivalen dengan 10-second rule x 6)
            hr_avg = round((len(peaks) / duration_sec) * 60)
            
            # 3. Hitung interval R-R dalam satuan detik
            rr_intervals_sec = np.diff(peaks) / TARGET_FS
            
            # Filter interval R-R untuk membuang noise yang tidak masuk akal (0.24 detik hingga 3.0 detik)
            valid_rr = [rr for rr in rr_intervals_sec if 0.24 <= rr <= 3.0]
            
            if len(valid_rr) >= 3:
                # Standar Klinis Holter (AAMI EC38):
                # Menghitung Max HR & Min HR menggunakan persentil 5 & 95 untuk mencegah 1 artefak sesaat menjadi 248 BPM
                p05_rr = np.percentile(valid_rr, 5)
                p95_rr = np.percentile(valid_rr, 95)
                hr_max = int(round(60 / p05_rr))
                hr_min = int(round(60 / p95_rr))
            elif len(valid_rr) > 0:
                min_rr = np.min(valid_rr)
                max_rr = np.max(valid_rr)
                hr_max = int(round(60 / min_rr))
                hr_min = int(round(60 / max_rr))
            else:
                hr_min = hr_max = hr_avg
            
            # Batasi nilai ekstrem agar tetap masuk akal secara klinis
            hr_min = max(30, min(hr_min, hr_avg))
            hr_max = min(250, max(hr_max, hr_avg))
        else:
            hr_avg = hr_min = hr_max = 75
            
        # --- Signal Quality Assessment (SQA) Lapis 2: Extreme Noise ---
        if hr_avg < 30 or hr_avg > 250:
            print(f"SQA Failed: Extreme noise detected (BPM: {hr_avg}).")
            try:
                urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:5001/api/sessions/{session_id}/ai-webhook", 
                    data=json.dumps({"error": f"Sinyal Sangat Buruk / Penuh Noise (Estimasi: {hr_avg} BPM). Pastikan kabel tidak bergoyang dan pasien diam. Analisis AI dibatalkan."}).encode('utf-8'), 
                    headers={'Content-Type': 'application/json'}
                ))
            except:
                pass
            return
        
        segments = []
        for i in range(0, len(signal_filtered) - WINDOW_SIZE, WINDOW_SIZE):
            segments.append(signal_filtered[i : i + WINDOW_SIZE])
            
        if len(segments) == 0:
            print("Signal too short")
            try:
                urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:5001/api/sessions/{session_id}/ai-webhook", data=json.dumps({"error": "Durasi rekaman terlalu pendek. Diperlukan minimal 10 detik data yang valid untuk dapat dianalisis AI."}).encode('utf-8'), headers={'Content-Type': 'application/json'}))
            except:
                pass
            return
            
        X = np.array(segments)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Gunakan model global secara thread-safe (Direct Call) tanpa memori tambahan
        print("Predicting...")
        predictions_tensor = model(X, training=False)
        predictions = predictions_tensor.numpy()
        
        episodes = []
        total_af_windows = 0
        
        window_predictions = []
        meta_file = os.path.join(BASE_DIR, 'results', 'final_model_metadata.json')
        calibrated_threshold = 0.5763
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    calibrated_threshold = float(json.load(f).get('median_optimal_threshold', 0.5763))
            except Exception:
                pass
        AF_THRESHOLD = calibrated_threshold
        for i, pred in enumerate(predictions):
            segment = X[i].flatten()
            pred_score = float(pred[0])
            is_af = int(pred_score >= AF_THRESHOLD)
            
            # --- CLINICAL REGULARITY GUARD: Tachycardia Regularity Check ---
            # Jika AI mendeteksi potensi AFib namun irama terbukti SANGAT TERATUR (CV_RR < 8%),
            # maka ini adalah Sinus Takikardia Normal (bukan Atrial Fibrilasi).
            if is_af:
                th_peak = np.max(segment) * 0.4
                if th_peak > 0.1:
                    peaks, _ = find_peaks(segment, height=th_peak, distance=30)
                    if len(peaks) >= 4:
                        rr_intervals = np.diff(peaks)
                        cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8)
                        if cv < 0.08:
                            is_af = False
                            pred_score = 0.10
            
            # --- HYBRID FALLBACK: Bradycardic AFib untuk Laporan ---
            if not is_af:
                threshold = np.max(segment) * 0.5
                if threshold > 0.1:
                    peaks, _ = find_peaks(segment, height=threshold, distance=60)
                    if 3 <= len(peaks) <= 8:
                        rr_intervals = np.diff(peaks)
                        mean_rr = np.mean(rr_intervals)
                        cv = np.std(rr_intervals) / (mean_rr + 1e-8)
                        if cv > 0.15:
                            is_af = True
                            pred_score = min(0.85 + cv, 0.99)
            
            window_predictions.append((i, is_af, float(pred_score)))

        # Kelompokkan jendela AF positif menjadi episode klinis (syarat episode AF: >= 30 detik / >= 3 jendela berturut-turut)
        episodes = []
        current_run = []
        total_af_windows = 0
        
        for w_idx, w_is_af, w_score in window_predictions:
            if w_is_af:
                total_af_windows += 1
                current_run.append((w_idx, w_score))
            else:
                if len(current_run) >= 3: # Minimal 3 jendela berturut-turut (>= 30 detik)
                    start_sec = current_run[0][0] * WINDOW_SEC
                    end_sec = (current_run[-1][0] + 1) * WINDOW_SEC
                    dur_sec = len(current_run) * WINDOW_SEC
                    startMin = str(int(start_sec // 60)).zfill(2)
                    startSec = str(int(start_sec % 60)).zfill(2)
                    endMin = str(int(end_sec // 60)).zfill(2)
                    endSec = str(int(end_sec % 60)).zfill(2)
                    episodes.append({
                        'id': len(episodes) + 1,
                        'start': f"{startMin}:{startSec}",
                        'end': f"{endMin}:{endSec}",
                        'duration': f"{dur_sec} detik",
                        'windows_count': len(current_run),
                        'mean_score': round(float(np.mean([s for _, s in current_run])), 4)
                    })
                current_run = []
                
        if len(current_run) >= 3:
            start_sec = current_run[0][0] * WINDOW_SEC
            end_sec = (current_run[-1][0] + 1) * WINDOW_SEC
            dur_sec = len(current_run) * WINDOW_SEC
            startMin = str(int(start_sec // 60)).zfill(2)
            startSec = str(int(start_sec % 60)).zfill(2)
            endMin = str(int(end_sec // 60)).zfill(2)
            endSec = str(int(end_sec % 60)).zfill(2)
            episodes.append({
                'id': len(episodes) + 1,
                'start': f"{startMin}:{startSec}",
                'end': f"{endMin}:{endSec}",
                'duration': f"{dur_sec} detik",
                'windows_count': len(current_run),
                'mean_score': round(float(np.mean([s for _, s in current_run])), 4)
            })

        total_analyzed_sec = len(segments) * WINDOW_SEC
        af_duration_sec = sum(ep['windows_count'] * WINDOW_SEC for ep in episodes)
        is_af_detected = len(episodes) > 0 # Terdeteksi hanya jika ada episode klinis valid (>= 30 detik)
        
        report_data = {
            'analysis_duration': round(total_analyzed_sec / 60, 2),
            'predicted_by': 'Model CNN-LSTM Hybrid AI (Background Job)',
            'is_af': is_af_detected,
            'overall_status': 'Episode Atrial Fibrillation Terdeteksi (durasi >= 30 detik)' if is_af_detected else 'AF Tidak Terdeteksi',
            'normal_rhythm': round((total_analyzed_sec - af_duration_sec) / 60, 2),
            'af_duration': round(af_duration_sec / 60, 2),
            'af_percentage': round((af_duration_sec / max(1, total_analyzed_sec)) * 100, 2),
            'hr_min': hr_min,
            'hr_avg': hr_avg,
            'hr_max': hr_max,
            'af_episodes': len(episodes),
            'episodes': episodes,
            'clinical_disclaimer': 'Prototipe riset akademis, bukan alat diagnosis medis resmi (belum berizin edar).'
        }
        
        webhook_url = f"http://127.0.0.1:5001/api/sessions/{session_id}/ai-webhook"
        data_bytes = json.dumps(report_data).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=data_bytes, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req)
        print(f"Background task finished. Webhook sent.")
        
    except Exception as e:
        print(f"Background task error: {e}")
        try:
            webhook_url = f"http://127.0.0.1:5001/api/sessions/{session_id}/ai-webhook"
            data_bytes = json.dumps({"error": f"Internal Error Python: {str(e)}"}).encode('utf-8')
            req = urllib.request.Request(webhook_url, data=data_bytes, headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req)
        except:
            pass

@app.route('/process_background', methods=['POST'])
def process_background():
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session_id provided'}), 400
        
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    thread = threading.Thread(target=background_task, args=(session_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Background processing started'})

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.json
    signal = data.get('signal', [])
    
    if len(signal) != 2500:
        return jsonify({"error": f"Need exactly 2500 points, got {len(signal)}"}), 400
        
    signal_np = np.array(signal)
    
    # Resample dari 250Hz ke 128Hz agar sesuai dengan model yang dilatih
    INPUT_FS = 250
    TARGET_FS = 128
    signal_resampled = resample_poly(signal_np, TARGET_FS, INPUT_FS)
    
    # Preprocessing
    signal_filtered = apply_bandpass_filter(signal_resampled, TARGET_FS)
    
    if np.std(signal_filtered) < 0.001:
        return jsonify({"is_af": False, "confidence": 0.0, "status": "flatline"})
        
    X = signal_filtered.reshape(1, 1280, 1)
    
    # Predict
    pred_tensor = model(X, training=False)
    pred_val = float(pred_tensor.numpy()[0][0])
    
    AF_THRESHOLD = 0.85
    is_af = bool(pred_val >= AF_THRESHOLD)
    
    # --- CLINICAL REGULARITY GUARD: Tachycardia Regularity Check ---
    # Jika AI mendeteksi potensi AFib namun irama terbukti SANGAT TERATUR (CV_RR < 8%),
    # maka ini adalah Sinus Takikardia Normal (bukan Atrial Fibrilasi).
    if is_af:
        th_peak = np.max(signal_filtered) * 0.4
        if th_peak > 0.1:
            peaks, _ = find_peaks(signal_filtered, height=th_peak, distance=30)
            if len(peaks) >= 4:
                rr_intervals = np.diff(peaks)
                cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8)
                if cv < 0.08:
                    is_af = False
                    pred_val = 0.10
    
    # --- HYBRID FALLBACK: Bradycardic AFib (BPM Rendah) ---
    if not is_af:
        # Gunakan threshold dinamis berdasarkan max amplitudo sinyal untuk menangkap puncak R
        threshold = np.max(signal_filtered) * 0.5
        
        if threshold > 0.1: # Pastikan sinyal bukan sekadar noise
            peaks, _ = find_peaks(signal_filtered, height=threshold, distance=60)
            
            # Jika hanya ada 3 hingga 8 detak dalam 10 detik (sekitar 18 - 48 BPM)
            if 3 <= len(peaks) <= 8:
                rr_intervals = np.diff(peaks)
                mean_rr = np.mean(rr_intervals)
                
                # Hitung Coefficient of Variation (CV) = Standar Deviasi / Rata-rata
                cv = np.std(rr_intervals) / (mean_rr + 1e-8)
                
                # Jika variasi jarak antar detak lebih dari 15%, ini sangat tidak normal (Fibrilasi)
                if cv > 0.15:
                    is_af = True
                    # Override pred_val dengan confidence tinggi
                    pred_val = min(0.85 + cv, 0.99)
    
    return jsonify({
        "is_af": is_af,
        "confidence": round(pred_val, 4) if is_af else round(1 - pred_val, 4),
        "status": "success"
    })

if __name__ == '__main__':
    print("API Machine Learning Berjalan di http://localhost:5003")
    app.run(host='0.0.0.0', port=5003)
