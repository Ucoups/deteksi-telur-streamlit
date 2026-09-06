import streamlit as st
import time
import pandas as pd
import os
import json
from utils.ui_styling import apply_custom_css
from utils.database import sync_manual_files, get_dataset_records

# Cek Login & Role Admin
if not st.session_state.get('logged_in', False):
    st.set_page_config(page_title="Train AI", page_icon="🧠", layout="wide")
    apply_custom_css()
    st.title("🧠 Pelatihan Model AI")
    st.warning("🔒 Silakan login terlebih dahulu di halaman Beranda!")
    st.stop()

if st.session_state.get('role') != 'admin':
    st.set_page_config(page_title="Train AI", page_icon="🧠", layout="wide")
    apply_custom_css()
    st.title("🧠 Pelatihan Model AI")
    st.error("🚫 Akses Ditolak: Halaman ini hanya dapat diakses oleh Admin!")
    st.stop()

st.set_page_config(page_title="Train AI", page_icon="🧠", layout="wide")
apply_custom_css()


def render_confusion_matrix(cm, class_names):
    # Cari index SEGAR dan BUSUK secara dinamis
    idx_segar = class_names.index("SEGAR") if "SEGAR" in class_names else 0
    idx_busuk = class_names.index("BUSUK") if "BUSUK" in class_names else 1
    
    tn = cm[idx_segar][idx_segar]  # Actual Segar, Pred Segar
    fp = cm[idx_segar][idx_busuk]  # Actual Segar, Pred Busuk
    fn = cm[idx_busuk][idx_segar]  # Actual Busuk, Pred Segar
    tp = cm[idx_busuk][idx_busuk]  # Actual Busuk, Pred Busuk
    
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total * 100 if total > 0 else 0.0
    
    # Render table HTML dengan CSS Neobrutalism
    cm_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; margin: 20px 0;">
      <table style="border: 3px solid black; border-collapse: collapse; text-align: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; box-shadow: 6px 6px 0px 0px #000000; width: 100%; max-width: 500px; background-color: white;">
        <thead>
          <tr style="background-color: #facc15; border-bottom: 3px solid black;">
            <th style="padding: 12px; border-right: 3px solid black; width: 34%;"></th>
            <th style="padding: 12px; border-right: 3px solid black; font-weight: 800; color: black; width: 33%;">Prediksi SEGAR</th>
            <th style="padding: 12px; font-weight: 800; color: black; width: 33%;">Prediksi BUSUK</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom: 3px solid black;">
            <td style="padding: 12px; border-right: 3px solid black; font-weight: 800; background-color: #f3f4f6; color: black;">Aktual SEGAR</td>
            <td style="padding: 25px; border-right: 3px solid black; background-color: #d1fae5; color: #065f46; font-size: 1.5rem; font-weight: 800;">{tn}</td>
            <td style="padding: 25px; background-color: #fee2e2; color: #991b1b; font-size: 1.5rem; font-weight: 800;">{fp}</td>
          </tr>
          <tr>
            <td style="padding: 12px; border-right: 3px solid black; font-weight: 800; background-color: #f3f4f6; color: black;">Aktual BUSUK</td>
            <td style="padding: 25px; border-right: 3px solid black; background-color: #fee2e2; color: #991b1b; font-size: 1.5rem; font-weight: 800;">{fn}</td>
            <td style="padding: 25px; background-color: #d1fae5; color: #065f46; font-size: 1.5rem; font-weight: 800;">{tp}</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top: 15px; font-weight: bold; font-size: 1.1rem; border: 3px solid black; padding: 10px; background-color: #facc15; box-shadow: 4px 4px 0px 0px #000000; color: black;">
        Akurasi Evaluasi Test Set: {accuracy:.2f}%
      </div>
    </div>
    """
    st.markdown(cm_html, unsafe_allow_html=True)

st.title("🧠 Pelatihan Model AI (Train AI)")
st.write("Latih jaringan saraf buatan (CNN) Anda menggunakan data gambar telur yang telah dikumpulkan.")

# Bagian 1: Sinkronisasi Data Manual
st.subheader("1. Sinkronisasi Data Manual")
st.write("Jika Anda menyalin gambar langsung ke folder komputer secara manual (tanpa lewat sistem web), tekan tombol di bawah agar terbaca oleh database.")

if st.button("🔄 Sinkronkan Folder Dataset", use_container_width=True):
    with st.spinner("Memindai folder dataset..."):
        added, total = sync_manual_files()
        time.sleep(1) # Efek UX
        st.success(f"Berhasil! {added} gambar baru didaftarkan. Total gambar siap dilatih: {total}.")

st.divider()

# Cek apakah ada data di database
records = get_dataset_records()
if len(records) < 10:
    st.warning("⚠️ Perhatian: Data di database Anda sangat sedikit. AI membutuhkan setidaknya 20-50 gambar per label untuk bisa belajar dengan baik.")

# Bagian 2: Pengaturan Pelatihan Model
st.subheader("2. Pengaturan Training Model")
col1, col2, col3 = st.columns(3)
with col1:
    model_choice = st.selectbox("Arsitektur Model AI", ["MobileNetV2", "ResNet18", "SqueezeNet"], index=0)
    st.caption("Pilih struktur saraf tiruan (CNN) yang ingin dilatih.")
with col2:
    epochs = st.slider("Jumlah Epoch (Perulangan Belajar)", min_value=1, max_value=20, value=3)
    st.caption("Semakin besar nilai Epoch, AI semakin pintar, namun waktu tunggu semakin lama.")
with col3:
    batch_size = st.selectbox("Ukuran Batch (Gambar per putaran)", [2, 4, 8, 16], index=1)
    st.caption("Tergantung kekuatan CPU Anda. Jika lemot/crash, gunakan 2 atau 4.")

if st.button("🚀 MULAI PELATIHAN AI (TRAIN MODEL)", use_container_width=True, type="primary"):
    from utils.ml_trainer import train_model_stream
    
    st.info("Proses *training* akan memakan resource CPU. Harap tunggu dan jangan tutup halaman ini.")
    
    # Progress bar containers
    progress_bar = st.progress(0, text="Mempersiapkan data...")
    status_text = st.empty()
    metric_cols = st.columns(2)
    acc_metric = metric_cols[0].empty()
    loss_metric = metric_cols[1].empty()
    
    st.subheader("📈 Kurva Pelatihan (Real-time)")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.write("**Akurasi (Train vs Test)**")
        acc_chart = st.empty()
    with chart_cols[1]:
        st.write("**Error / Loss (Train vs Test)**")
        loss_chart = st.empty()
        
    log_container = st.container()
    
    from utils.config import IMAGE_DIR
    data_dir = IMAGE_DIR
    
    # Dynamic metric lists for live plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Menjalankan generator training
    for step in train_model_stream(data_dir, model_name=model_choice, epochs=epochs, batch_size=batch_size):
        if step["status"] == "info":
            status_text.write(f"ℹ️ {step['message']}")
            
        elif step["status"] == "error":
            st.error(step['message'])
            break
            
        elif step["status"] == "progress_batch":
            # Kalkulasi overall progress percentage
            current_epoch = step["epoch"]
            current_batch = step["batch"]
            total_batches = step["total_batches"]
            
            # Base progress per epoch
            base_prog = (current_epoch - 1) / epochs
            # Progress dalam epoch saat ini
            epoch_prog = (current_batch / total_batches) * (1 / epochs)
            
            total_prog = int((base_prog + epoch_prog) * 100)
            progress_bar.progress(total_prog, text=f"Epoch {current_epoch}/{epochs} - Batch {current_batch}/{total_batches}...")
            
        elif step["status"] == "progress_epoch":
            train_losses.append(step['loss'])
            train_accs.append(step['accuracy'])
            val_losses.append(step['val_loss'])
            val_accs.append(step['val_accuracy'])
            
            # Update metrics
            acc_metric.metric("Akurasi Latih", f"{step['accuracy']:.2f}%", f"Test: {step['val_accuracy']:.2f}%")
            loss_metric.metric("Error Latih (Loss)", f"{step['loss']:.4f}", f"Test: {step['val_loss']:.4f}", delta_color="inverse")
            
            log_container.success(
                f"✅ Selesai Epoch {step['epoch']} | "
                f"Latih Acc: {step['accuracy']:.2f}% (Loss: {step['loss']:.4f}) | "
                f"Test Acc: {step['val_accuracy']:.2f}% (Loss: {step['val_loss']:.4f})"
            )
            
            # Update charts
            epochs_range = list(range(1, len(train_losses) + 1))
            
            df_acc = pd.DataFrame({
                'Epoch': epochs_range,
                'Train Accuracy': train_accs,
                'Test Accuracy': val_accs
            }).set_index('Epoch')
            acc_chart.line_chart(df_acc)
            
            df_loss = pd.DataFrame({
                'Epoch': epochs_range,
                'Train Loss': train_losses,
                'Test Loss': val_losses
            }).set_index('Epoch')
            loss_chart.line_chart(df_loss)
            
        elif step["status"] == "done":
            progress_bar.progress(100, text="Pelatihan Selesai!")
            st.balloons()
            st.success(f"🎉 Selesai! {step['message']}")
            
            # Tampilkan confusion matrix final
            st.write("### 🎚️ Confusion Matrix Final")
            render_confusion_matrix(step["metrics"]["confusion_matrix"], step["metrics"]["class_names"])
            
            st.write("Sekarang Anda bisa mencoba model baru ini di halaman **📷 Deteksi Live**!")

# Bagian 3: Evaluasi Model Terakhir (Persistent)
metrics_path = 'models/evaluation_metrics.json'
if os.path.exists(metrics_path):
    st.divider()
    st.subheader("📊 Hasil Evaluasi Model Saat Ini")
    st.write("Berikut adalah hasil performa dari model AI yang terakhir kali berhasil dilatih.")
    
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
            
        # Tampilkan timestamp
        if 'timestamp' in metrics_data:
            st.caption(f"Pelatihan terakhir diselesaikan pada: **{metrics_data['timestamp']}**")
            
        # Buat grafik Kurva Evaluasi
        st.write("### 📈 Kurva Pelatihan")
        col_curve1, col_curve2 = st.columns(2)
        
        epochs_count = len(metrics_data['train_loss'])
        epochs_range = list(range(1, epochs_count + 1))
        
        with col_curve1:
            st.write("**Akurasi (Train vs Test)**")
            df_acc = pd.DataFrame({
                'Epoch': epochs_range,
                'Train Accuracy': metrics_data['train_acc'],
                'Test Accuracy': metrics_data['val_acc']
            }).set_index('Epoch')
            st.line_chart(df_acc)
            
        with col_curve2:
            st.write("**Error / Loss (Train vs Test)**")
            df_loss = pd.DataFrame({
                'Epoch': epochs_range,
                'Train Loss': metrics_data['train_loss'],
                'Test Loss': metrics_data['val_loss']
            }).set_index('Epoch')
            st.line_chart(df_loss)
            
        # Tampilkan Confusion Matrix
        st.write("### 🎚️ Confusion Matrix")
        render_confusion_matrix(metrics_data['confusion_matrix'], metrics_data['class_names'])
        
    except Exception as e:
        st.warning(f"Gagal memuat metrik evaluasi: {str(e)}")

