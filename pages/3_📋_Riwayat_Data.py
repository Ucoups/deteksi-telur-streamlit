import streamlit as st
import pandas as pd
from utils.ui_styling import apply_custom_css
from utils.database import get_detection_history

# Cek Login
if not st.session_state.get('logged_in', False):
    st.set_page_config(page_title="Riwayat Data", page_icon="📋", layout="wide")
    apply_custom_css()
    st.title("📋 Riwayat Deteksi")
    st.warning("🔒 Silakan login terlebih dahulu di halaman Beranda!")
    st.stop()

st.set_page_config(page_title="Riwayat Data", page_icon="📋", layout="wide")
apply_custom_css()


st.title("📋 Riwayat Deteksi")
st.write("Log audit operasional pengecekan telur yang direkam oleh sistem.")

# Ambil data riwayat dari database (100 log terbaru)
history_records = get_detection_history(limit=100)

if not history_records:
    st.info("Belum ada riwayat deteksi. Data akan otomatis muncul setelah Anda mengecek telur di menu Deteksi Live.")
else:
    # Ubah format array tuple ke DataFrame pandas
    df = pd.DataFrame(history_records, columns=["Waktu Deteksi", "Operator / Sistem", "Hasil Klasifikasi", "Tingkat Keyakinan"])
    
    # Format nilai Akurasi menjadi persentase agar lebih enak dibaca
    df["Tingkat Keyakinan"] = df["Tingkat Keyakinan"].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(df, use_container_width=True)
    
    # Ekspor Laporan
    st.divider()
    st.subheader("📥 Ekspor Laporan")
    st.write("Unduh data riwayat deteksi dalam format CSV untuk keperluan administrasi atau analisis lebih lanjut.")
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Laporan Riwayat (CSV)",
        data=csv_data,
        file_name=f"riwayat_deteksi_telur_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

