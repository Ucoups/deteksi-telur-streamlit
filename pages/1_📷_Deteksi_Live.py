import streamlit as st
import time
import numpy as np
from PIL import Image
import io
import os
from utils.ui_styling import apply_custom_css
from utils.ml_inference import predict_egg, load_model

# Cek Login
if not st.session_state.get('logged_in', False):
    st.set_page_config(page_title="Deteksi Live - EggQuality AI", page_icon="📷", layout="wide")
    apply_custom_css()
    st.title("📷 Deteksi Kualitas Telur")
    st.warning("🔒 Silakan login terlebih dahulu di halaman Beranda!")
    st.stop()

st.set_page_config(page_title="Deteksi Live - EggQuality AI", page_icon="📷", layout="wide")
apply_custom_css()

st.title("📷 Deteksi Kualitas Telur")
st.caption("Didukung oleh Explainable AI (Grad-CAM) & Out-of-Distribution Detection")

# ─── Input Gambar ──────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("① Input Gambar Telur")
    source = st.radio("Pilih Sumber:", ["📂 Upload File", "📸 Kamera", "🤖 Simulasi IoT Pabrik"], horizontal=True)

    img_file = None
    if source == "📂 Upload File":
        img_file = st.file_uploader(
            "Unggah gambar telur (JPG/PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Pastikan gambar menampilkan telur secara jelas."
        )
    elif source == "📸 Kamera":
        img_file = st.camera_input("Ambil foto dari kamera")
        
    elif source == "🤖 Simulasi IoT Pabrik":
        st.write("🏭 **Mode Pemindaian Otomatis (Conveyor Belt Pabrik)**")
        st.caption("AI akan mensimulasikan kamera sensor pada ban berjalan pabrik telur, memproses gambar secara kontinu.")
        col_sim_ctrl1, col_sim_ctrl2 = st.columns(2)
        with col_sim_ctrl1:
            run_sim = st.toggle("▶️ Jalankan Konveyor Pabrik", value=False)
        with col_sim_ctrl2:
            sim_speed = st.slider("Kecepatan Pemrosesan (detik)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    if img_file and source != "🤖 Simulasi IoT Pabrik":
        st.image(img_file, caption="Gambar yang akan dianalisis", use_container_width=True)

# ─── Tombol Deteksi ───────────────────────────────────────────────────────────
if img_file and source != "🤖 Simulasi IoT Pabrik":
    st.divider()
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
    with btn_col1:
        if source == "📸 Kamera":
            run_detection = True
            st.success("📸 Gambar ditangkap! Memproses otomatis...")
        else:
            run_detection = st.button("🚀 Mulai Deteksi AI", use_container_width=True, type="primary")

    # Toggle Grad-CAM dan OOD
    with btn_col2:
        show_gradcam = st.toggle("🔥 Tampilkan Grad-CAM", value=True)
    with btn_col3:
        show_ood_detail = st.toggle("🛡️ Detail OOD", value=False)

# ─── Proses Deteksi ───────────────────────────────────────────────────────────
with col_result:
    st.subheader("② Hasil Analisis AI")

    if source != "🤖 Simulasi IoT Pabrik":
        if img_file and run_detection:
            img_bytes = img_file.getvalue()

            with st.spinner("🔍 Menganalisis gambar..."):
                model, class_names = load_model()

                if model is None:
                    st.error("❌ **Model AI belum dilatih!**")
                    st.info("Masuk ke menu **🧠 Train AI** dan latih model terlebih dahulu.")
                    st.stop()

                # ─── LAYER 1: OOD DETECTION ────────────────────────────────
                from utils.ood_detector import detect_ood
                is_ood, ood_reason, ood_metrics = detect_ood(img_bytes, model)

                if is_ood:
                    st.error("🚫 **Gambar Ditolak: Bukan Telur Ayam**")
                    st.warning(ood_reason)

                    if show_ood_detail:
                        st.subheader("📊 Detail Pemeriksaan OOD")
                        c1, c2 = st.columns(2)
                        c1.metric(
                            "Confidence AI",
                            f"{ood_metrics['max_confidence']}%",
                            f"Threshold: ≥ {ood_metrics['threshold_msp']}%",
                            delta_color="inverse"
                        )
                        c2.metric(
                            "Free Energy (Stabilitas)",
                            f"{ood_metrics['free_energy']}",
                            f"Threshold: ≤ {ood_metrics['threshold_energy']}",
                            delta_color="inverse"
                        )
                        st.caption(
                            "**Interpretasi Free Energy:** Nilai negatif besar = Dikenali kuat. "
                            "Mendekati 0 atau positif = Asing / Out-of-Distribution."
                        )

                else:
                    # ─── LAYER 2: KLASIFIKASI UTAMA ────────────────────────
                    result, confidence = predict_egg(img_bytes)

                    if result == "SEGAR":
                        st.success(f"✅ **TELUR SEGAR**")
                        st.progress(float(confidence), text=f"Tingkat Keyakinan AI: {confidence*100:.1f}%")
                    elif result == "BUSUK":
                        st.error(f"❌ **TELUR BUSUK**")
                        st.progress(float(confidence), text=f"Tingkat Keyakinan AI: {confidence*100:.1f}%")
                    else:
                        st.info(f"ℹ️ **{result}**")
                        st.progress(float(confidence), text=f"Tingkat Keyakinan AI: {confidence*100:.1f}%")

                    # Detail OOD (jika toggle aktif)
                    if show_ood_detail:
                        with st.expander("📊 Detail Pemeriksaan OOD (Lolos)"):
                            c1, c2 = st.columns(2)
                            c1.metric("Confidence AI", f"{ood_metrics['max_confidence']}%", "✅ Lolos MSP")
                            c2.metric("Free Energy", f"{ood_metrics['free_energy']}", "✅ Lolos Energy Check")

                    # Simpan log dengan operator aktif
                    from utils.database import save_log
                    operator_name = st.session_state.get('name', 'Operator')
                    save_log(operator_name, result, confidence)

                    # ─── LAYER 3: GRAD-CAM VISUALIZATION ──────────────────
                    if show_gradcam:
                        st.divider()
                        st.subheader("🔥 Explainable AI — Grad-CAM")
                        st.caption(
                            "Peta panas ini menunjukkan **area gambar** yang paling diperhatikan AI "
                            "dalam membuat keputusan. Warna **merah/kuning = area paling berpengaruh**."
                        )

                        try:
                            from utils.gradcam import apply_gradcam_overlay

                            # Tentukan class index berdasarkan prediksi
                            if class_names and result in [c.upper() for c in class_names]:
                                class_idx = [c.upper() for c in class_names].index(result)
                            else:
                                class_idx = None

                            with st.spinner("🎨 Membuat visualisasi Grad-CAM..."):
                                overlay_img, heatmap_img = apply_gradcam_overlay(
                                    img_bytes, model, class_idx=class_idx, alpha=0.45
                                )

                            # Tampilkan side-by-side
                            gc1, gc2 = st.columns(2)
                            with gc1:
                                st.image(
                                    overlay_img,
                                    caption="🖼️ Overlay Grad-CAM (Original + Heatmap)",
                                    use_container_width=True,
                                    clamp=True
                                )
                            with gc2:
                                st.image(
                                    heatmap_img,
                                    caption="🌡️ Peta Panas Murni (Semakin Merah = Lebih Penting)",
                                    use_container_width=True,
                                    clamp=True
                                )

                            st.info(
                                "💡 **Cara Membaca:** Area berwarna **merah/jingga** adalah bagian cangkang telur "
                                "yang paling menentukan keputusan AI (misal: bintik, warna tidak merata, retakan). "
                                "Area **biru/ungu** tidak terlalu berpengaruh pada keputusan."
                            )

                        except Exception as e:
                            st.warning(f"⚠️ Grad-CAM tidak dapat ditampilkan: {str(e)}")

        elif img_file:
            st.info("👈 Silakan klik **🚀 Mulai Deteksi AI** di bawah untuk menganalisis gambar ini.")
        else:
            st.info("👈 Silakan unggah atau ambil foto telur terlebih dahulu.")

    else:
        # Simulasi IoT Conveyor Pabrik
        if run_sim:
            from utils.database import get_dataset_records
            records = get_dataset_records()
            if not records:
                st.warning("Dataset kosong. Silakan unggah gambar ke dataset terlebih dahulu.")
            else:
                placeholder_img = st.empty()
                placeholder_result = st.empty()
                placeholder_stats = st.empty()

                model, class_names = load_model()
                if model is None:
                    st.error("❌ **Model AI belum dilatih!**")
                    st.info("Masuk ke menu **🧠 Train AI** dan latih model terlebih dahulu.")
                    st.stop()

                segar_count = 0
                busuk_count = 0
                import random
                from utils.config import IMAGE_DIR

                try:
                    for _ in range(50):
                        # Ambil gambar acak dari dataset
                        rec = random.choice(records)
                        filename, label = rec[1], rec[2]
                        img_path = os.path.join(IMAGE_DIR, label.lower(), filename)

                        if os.path.exists(img_path):
                            with open(img_path, "rb") as f:
                                img_bytes = f.read()

                            # Tampilkan gambar
                            img_pil = Image.open(io.BytesIO(img_bytes))
                            placeholder_img.image(img_pil, caption=f"Scanner Konveyor: {filename}", use_container_width=True)

                            # Jalankan prediksi
                            result, confidence = predict_egg(img_bytes)

                            # Simpan log audit pabrik
                            from utils.database import save_log
                            operator_name = f"IoT_Conveyor ({st.session_state.get('name', 'Operator')})"
                            save_log(operator_name, result, confidence)

                            if result == "SEGAR":
                                segar_count += 1
                                placeholder_result.markdown(
                                    f'<div style="border: 3px solid black; padding: 15px; background-color: #d1fae5; color: #065f46; font-weight: 800; font-size: 1.5rem; text-align: center; box-shadow: 4px 4px 0px black;">🟢 TELUR SEGAR ({confidence*100:.1f}%)</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                busuk_count += 1
                                placeholder_result.markdown(
                                    f'<div style="border: 3px solid black; padding: 15px; background-color: #fee2e2; color: #991b1b; font-weight: 800; font-size: 1.5rem; text-align: center; box-shadow: 4px 4px 0px black;">🔴 TELUR BUSUK ({confidence*100:.1f}%)</div>',
                                    unsafe_allow_html=True
                                )

                            # Update statistik live
                            total_uji = segar_count + busuk_count
                            segar_pct = (segar_count / total_uji * 100) if total_uji > 0 else 0
                            busuk_pct = (busuk_count / total_uji * 100) if total_uji > 0 else 0
                            
                            placeholder_stats.markdown(
                                f"""
                                <div style="border: 3px solid black; padding: 15px; margin-top: 15px; background-color: #facc15; box-shadow: 4px 4px 0px black; color: black; font-weight: bold;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 1.2rem;">
                                      <span>📊 Total Diuji: {total_uji}</span>
                                      <span style="color: #065f46;">🟢 Segar: {segar_count} ({segar_pct:.1f}%)</span>
                                      <span style="color: #991b1b;">🔴 Busuk: {busuk_count} ({busuk_pct:.1f}%)</span>
                                    </div>
                                    <div style="width: 100%; background-color: #fee2e2; border: 2px solid black; height: 24px; border-radius: 12px; overflow: hidden; display: flex;">
                                        <div style="width: {segar_pct}%; background-color: #22c55e; height: 100%;"></div>
                                        <div style="width: {busuk_pct}%; background-color: #ef4444; height: 100%;"></div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        time.sleep(sim_speed)
                except Exception as e:
                    pass
        else:
            st.info("Aktifkan toggle di sebelah kiri untuk memulai simulasi deteksi ban berjalan pabrik.")

# ─── Simpan ke Dataset (Khusus Admin) ──────────────────────────────────────────
if img_file and source != "🤖 Simulasi IoT Pabrik":
    if st.session_state.get('role') == 'admin':
        st.divider()
        with st.expander("💾 Simpan Gambar ke Dataset Pelatihan (Khusus Admin)"):
            st.write("Bantu AI menjadi lebih akurat! Tandai gambar ini dengan label yang benar.")
            col_a, col_b = st.columns(2)
            with col_a:
                label_koreksi = st.selectbox("Label Sebenarnya (Ground Truth):", ["SEGAR", "BUSUK"])
            with col_b:
                split_type = st.radio("Tipe Data:", ["Train", "Test"])

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📥 Simpan ke Dataset", use_container_width=True):
                    from utils.database import save_to_dataset
                    save_to_dataset(img_file.getvalue(), label_koreksi, split_type)
                    st.success(f"✅ Gambar disimpan sebagai **{label_koreksi}** untuk data **{split_type}**.")
                    
            with col_btn2:
                if st.button("🚀 Retrain Cepat AI (1 Epoch)", use_container_width=True, type="primary"):
                    from utils.database import save_to_dataset
                    save_to_dataset(img_file.getvalue(), label_koreksi, split_type)
                    
                    with st.spinner("🧠 AI sedang belajar dari gambar ini..."):
                        from utils.ml_trainer import train_quick_epoch
                        success, msg = train_quick_epoch(img_file.getvalue(), label_koreksi)
                        if success:
                            st.success(f"✅ {msg}")
                            st.balloons()
                        else:
                            st.error(f"❌ {msg}")

