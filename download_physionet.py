import os
import urllib.request
import urllib.error
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw_data')
LTAF_DIR = os.path.join(RAW_DIR, 'ltafdb')
SHDB_DIR = os.path.join(RAW_DIR, 'shdb-af')

os.makedirs(LTAF_DIR, exist_ok=True)
os.makedirs(SHDB_DIR, exist_ok=True)

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

def get_ltafdb_records():
    url = 'https://physionet.org/files/ltafdb/1.0.0/RECORDS'
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        content = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
        records = [line.strip() for line in content.splitlines() if line.strip()]
        return records
    except Exception as e:
        print(f"Error fetching ltafdb records list: {e}")
        return []

def get_shdb_records():
    url = 'https://physionet.org/files/shdb-af/1.0.1/'
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        html = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
        records = sorted(list(set(re.findall(r'href="([^"]+)\.hea"', html))))
        return records
    except Exception as e:
        print(f"Error fetching shdb-af records list: {e}")
        return []

def download_file(url, local_path, max_retries=5):
    # Jika file sudah ada dan ukurannya > 0, lewati
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return True, "sudah ada"
    
    tmp_path = local_path + ".tmp"
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_path, 'wb') as out_file:
                chunk_size = 256 * 1024 # 256 KB chunk
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            
            if os.path.exists(local_path):
                os.remove(local_path)
            os.rename(tmp_path, local_path)
            sz_mb = os.path.getsize(local_path) / (1024 * 1024)
            return True, f"sukses ({sz_mb:.2f} MB)"
        except Exception as e:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
            if attempt == max_retries:
                return False, f"gagal ({e})"
            time.sleep(2 * attempt)

def download_dataset(db_name, base_url, records, extensions, save_dir, max_workers=4):
    print("=================================================================")
    print(f" Memulai Pengunduhan Database: {db_name} ({len(records)} Rekaman)")
    print(f" Direktori Tujuan: {save_dir}")
    print("=================================================================")
    
    tasks = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rec in records:
            for ext in extensions:
                filename = f"{rec}.{ext}"
                url = f"{base_url}/{filename}"
                local_path = os.path.join(save_dir, filename)
                future = executor.submit(download_file, url, local_path)
                tasks[future] = filename
                
        completed = 0
        total = len(tasks)
        success_count = 0
        fail_count = 0
        
        for future in as_completed(tasks):
            completed += 1
            fname = tasks[future]
            ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"  [GAGAL] {fname}: {msg}")
                
            if completed % 10 == 0 or completed == total:
                pct = (completed / total) * 100
                print(f"[{db_name}] Selesai: {completed}/{total} file ({pct:.1f}%) | Sukses: {success_count}, Gagal: {fail_count}")

def main():
    print("Memulai proses pengunduhan dataset PhysioNet (ltafdb & shdb-af)...")
    
    # 1. Unduh Database ltafdb
    ltaf_records = get_ltafdb_records()
    print(f"Total rekaman ltafdb terdeteksi: {len(ltaf_records)}")
    if ltaf_records:
        download_dataset(
            db_name="ltafdb",
            base_url="https://physionet.org/files/ltafdb/1.0.0",
            records=ltaf_records,
            extensions=["hea", "dat", "atr"],
            save_dir=LTAF_DIR,
            max_workers=4
        )
    
    # 2. Unduh Database shdb-af
    shdb_records = get_shdb_records()
    print(f"Total rekaman shdb-af terdeteksi: {len(shdb_records)}")
    if shdb_records:
        download_dataset(
            db_name="shdb-af",
            base_url="https://physionet.org/files/shdb-af/1.0.1",
            records=shdb_records,
            extensions=["hea", "dat", "atr"],
            save_dir=SHDB_DIR,
            max_workers=4
        )
        
    print("\n=================================================================")
    print(" SELURUH REKAMAN DARI 2 DATABASE PHYSIONET TELAH BERHASIL DIUNDUH!")
    print("=================================================================")

if __name__ == '__main__':
    main()
