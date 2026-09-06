import streamlit as st
import pandas as pd
from utils.ui_styling import apply_custom_css
from utils.database import get_dataset_records

# Cek Login & Role Admin
if not st.session_state.get('logged_in', False):
    st.set_page_config(page_title="Dataset Manager", page_icon="📁", layout="wide")
    apply_custom_css()
    st.title("📁 Pengelola Dataset")
    st.warning("🔒 Silakan login terlebih dahulu di halaman Beranda!")
    st.stop()

if st.session_state.get('role') != 'admin':
    st.set_page_config(page_title="Dataset Manager", page_icon="📁", layout="wide")
    apply_custom_css()
    st.title("📁 Pengelola Dataset")
    st.error("🚫 Akses Ditolak: Halaman ini hanya dapat diakses oleh Admin!")
    st.stop()

st.set_page_config(page_title="Dataset Manager", page_icon="📁", layout="wide")
apply_custom_css()


st.title("📁 Pengelola Dataset (Train & Test)")
st.write("Di sini Anda bisa melihat daftar gambar yang telah disimpan untuk melatih ulang AI.")

st.divider()

with st.expander("🚀 Upload Banyak Gambar Sekaligus (Bulk Upload)", expanded=False):
    st.write("Gunakan fitur ini untuk mengunggah puluhan/ratusan gambar sekaligus ke dalam dataset Anda.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        bulk_label = st.selectbox("Pilih Label untuk Semua Gambar:", ["SEGAR", "BUSUK"], key="bulk_label")
    with col_b:
        bulk_split = st.radio("Pilih Tipe Data:", ["Train", "Test"], key="bulk_split")
        
    uploaded_files = st.file_uploader("Pilih gambar-gambar Anda (Bisa blok/pilih banyak file sekaligus)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if st.button("Mulai Proses Bulk Upload 📥", use_container_width=True):
        if uploaded_files:
            from utils.database import save_to_dataset
            
            progress_text = "Menyimpan gambar ke database. Mohon tunggu..."
            my_bar = st.progress(0, text=progress_text)
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                img_bytes = file.getvalue()
                save_to_dataset(img_bytes, bulk_label, bulk_split)
                
                # Update progress
                progress = int(((i + 1) / total_files) * 100)
                my_bar.progress(progress, text=f"Menyimpan file {i+1} dari {total_files}...")
                
            st.success(f"✅ Selesai! {total_files} gambar berhasil ditambahkan sebagai data **{bulk_split}** dengan label **{bulk_label}**.")
            
        else:
            st.error("Silakan unggah minimal satu gambar terlebih dahulu!")

st.divider()
st.subheader("Tabel Rekap Dataset Saat Ini")

# Ambil data dari database
records = get_dataset_records()

if len(records) == 0:
    st.info("Belum ada data gambar yang disimpan. Silakan simpan gambar dari halaman Deteksi Live.")
else:
    # Ubah format ke DataFrame
    df = pd.DataFrame(records, columns=["ID", "Nama File", "Label", "Tipe Split", "Waktu Simpan"])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Gambar", len(df))
    col2.metric("Data Train", len(df[df['Tipe Split'] == 'TRAIN']))
    col3.metric("Data Test", len(df[df['Tipe Split'] == 'TEST']))
    
    st.dataframe(df, use_container_width=True)
    
    st.write("💡 *Tip: File gambar fisik tersimpan di folder `database/dataset_images/segar/` dan `busuk/`.*")
