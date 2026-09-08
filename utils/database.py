import sqlite3
import os
import uuid
from typing import List, Tuple, Dict, Any
from utils.config import DB_PATH, IMAGE_DIR

def get_connection() -> sqlite3.Connection:
    """Membuka koneksi ke SQLite secara terpusat."""
    return sqlite3.connect(DB_PATH)

def init_db() -> None:
    """Inisialisasi tabel dan folder database jika belum ada."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_DIR, 'segar'), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_DIR, 'busuk'), exist_ok=True)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Tabel log harian/deteksi
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS riwayat_deteksi (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pegawai TEXT,
                hasil TEXT,
                akurasi REAL,
                waktu_deteksi TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabel pengumpulan dataset (Train/Test)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_training (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                label TEXT,
                dataset_split TEXT,
                waktu_simpan TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def save_log(pegawai: str, hasil: str, akurasi: float) -> None:
    """Simpan log ke database setiap kali selesai deteksi."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO riwayat_deteksi (pegawai, hasil, akurasi) VALUES (?, ?, ?)",
            (pegawai, hasil, akurasi)
        )
        conn.commit()

def clear_detection_history() -> None:
    """Menghapus seluruh data dari tabel riwayat_deteksi."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM riwayat_deteksi")
        conn.commit()

def save_to_dataset(image_bytes: bytes, label: str, split_type: str) -> str:
    """Menyimpan gambar fisik ke folder dan mencatatnya di SQLite."""
    init_db()
    
    ext = ".jpg"
    filename = f"{label.lower()}_{split_type.lower()}_{uuid.uuid4().hex[:8]}{ext}"
    
    target_dir = os.path.join(IMAGE_DIR, label.lower())
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename)
    
    with open(file_path, "wb") as f:
        f.write(image_bytes)
        
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO dataset_training (filename, label, dataset_split) VALUES (?, ?, ?)",
            (filename, label.upper(), split_type.upper())
        )
        conn.commit()
        
    return file_path

def get_dataset_records() -> List[Tuple]:
    """Mengambil seluruh log gambar dataset."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, label, dataset_split, waktu_simpan FROM dataset_training ORDER BY id DESC")
        data = cursor.fetchall()
    return data

def delete_dataset_by_id(item_id: int) -> bool:
    """Menghapus data dataset berdasarkan ID, termasuk file fisik."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, label FROM dataset_training WHERE id = ?", (item_id,))
        record = cursor.fetchone()
        
        if record:
            filename, label = record
            file_path = os.path.join(IMAGE_DIR, label.lower(), filename)
            
            # Hapus file fisik jika ada
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            
            # Hapus record dari DB
            cursor.execute("DELETE FROM dataset_training WHERE id = ?", (item_id,))
            conn.commit()
            return True
    return False

def clear_all_dataset() -> None:
    """Menghapus seluruh rekaman dataset dari DB dan menghapus file fisiknya."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, label FROM dataset_training")
        records = cursor.fetchall()
        
        for filename, label in records:
            file_path = os.path.join(IMAGE_DIR, label.lower(), filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                    
        cursor.execute("DELETE FROM dataset_training")
        conn.commit()

def sync_manual_files() -> Tuple[int, int]:
    """Memindai folder fisik untuk sinkronisasi gambar manual ke DB."""
    init_db()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM dataset_training")
        existing_files = {row[0] for row in cursor.fetchall()}
        
        added_count = 0
        labels = ['segar', 'busuk']
        
        for label in labels:
            folder_path = os.path.join(IMAGE_DIR, label)
            if not os.path.exists(folder_path):
                continue
                
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file not in existing_files:
                    cursor.execute(
                        "INSERT INTO dataset_training (filename, label, dataset_split) VALUES (?, ?, ?)",
                        (file, label.upper(), 'TRAIN')
                    )
                    added_count += 1
                    
        conn.commit()
        cursor.execute("SELECT COUNT(*) FROM dataset_training")
        total_count = cursor.fetchone()[0]
        
    return added_count, total_count

def get_detection_history(limit: int = 100) -> List[Tuple]:
    """Mengambil log audit operasional dari database."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT waktu_deteksi, pegawai, hasil, akurasi FROM riwayat_deteksi ORDER BY waktu_deteksi DESC LIMIT ?", 
            (limit,)
        )
        return cursor.fetchall()

def get_dashboard_stats() -> Dict[str, Any]:
    """Mengambil data statistik untuk dashboard berdasarkan DB log."""
    init_db()
    stats = {
        "total_uji": 0,
        "total_segar": 0,
        "total_busuk": 0,
        "today_uji": 0,
        "today_segar": 0,
        "today_busuk": 0,
        "delta_uji": 0,
        "delta_segar": 0,
        "delta_busuk": 0,
        "grafik": {}
    }
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM riwayat_deteksi")
        stats["total_uji"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM riwayat_deteksi WHERE hasil = 'SEGAR'")
        stats["total_segar"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM riwayat_deteksi WHERE hasil = 'BUSUK'")
        stats["total_busuk"] = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT date(waktu_deteksi) as d,
                   SUM(CASE WHEN hasil = 'SEGAR' THEN 1 ELSE 0 END) as segar_count,
                   SUM(CASE WHEN hasil = 'BUSUK' THEN 1 ELSE 0 END) as busuk_count
            FROM riwayat_deteksi
            GROUP BY d
            ORDER BY d DESC LIMIT 7
        ''')
        rows = cursor.fetchall()
        
        if len(rows) > 0:
            stats["today_segar"] = rows[0][1]
            stats["today_busuk"] = rows[0][2]
            stats["today_uji"] = stats["today_segar"] + stats["today_busuk"]
            
        if len(rows) > 1:
            yesterday_segar = rows[1][1]
            yesterday_busuk = rows[1][2]
            yesterday_uji = yesterday_segar + yesterday_busuk
            stats["delta_uji"] = stats["today_uji"] - yesterday_uji
            stats["delta_segar"] = stats["today_segar"] - yesterday_segar
            stats["delta_busuk"] = stats["today_busuk"] - yesterday_busuk
        else:
            stats["delta_uji"] = stats["today_uji"]
            stats["delta_segar"] = stats["today_segar"]
            stats["delta_busuk"] = stats["today_busuk"]
        
        for r in reversed(rows):
            stats["grafik"][r[0]] = {"Segar": r[1], "Busuk": r[2]}
            
    return stats
