# 🥚 EggQuality AI: Sistem Deteksi Kualitas Telur Cerdas Berbasis Machine Learning

**EggQuality AI** adalah sebuah aplikasi *web-based* cerdas yang dirancang untuk mengotomatisasi inspeksi kualitas telur (segar vs busuk) secara *real-time*. Dibangun menggunakan ekosistem **Python, Streamlit, dan PyTorch**, sistem ini ditujukan untuk industri peternakan maupun perlombaan teknologi dengan mengedepankan akurasi, keamanan, dan *Explainable AI* (XAI).

🌐 **Live Demo:** [https://deteksi-telurr.streamlit.app/](https://deteksi-telurr.streamlit.app/)

---

## ✨ Fitur Utama (Core Features)

### 1. 🔐 Role-Based Access Control (RBAC)
Sistem dilengkapi dengan autentikasi multi-level untuk menjaga keamanan data industri:
- **Administrator**: Memiliki akses penuh ke sistem klasifikasi, termasuk Hak Akses untuk melakukan *Training* Model AI, manajemen dataset, dan menghapus riwayat operasional.
- **Operator Pabrik**: Akses terbatas yang hanya diizinkan untuk menjalankan pemindaian telur *(Deteksi Live)*, memantau *Dashboard* harian, dan mengekspor laporan.

### 2. 📷 Deteksi Live & Simulasi IoT Ban Berjalan
Dilengkapi dengan fitur pemindaian menggunakan antarmuka kamera secara langsung *(Live Camera)* maupun unggah manual. Untuk mendemonstrasikan kapabilitas industri, terdapat fitur **Simulasi IoT Pabrik** yang mampu memindai puluhan telur secara konsekutif menyerupai sistem ban berjalan *(conveyor belt)* pada lini produksi.

### 3. 🧠 Explainable AI (Grad-CAM) & Energy-Based OOD
AI ini tidak beroperasi sebagai sekadar *Black Box* tradisional:
- **Grad-CAM**: Setiap prediksi akan disertai *Heatmap* (peta panas) visual. Model secara transparan menyoroti area spesifik (misal: bintik darah, tekstur cangkang retak) yang memicu keputusan *convolutional filters*.
- **Energy-Based Out-of-Distribution (OOD)**: Mengimplementasikan algoritma *Helmholtz Free Energy* (terinspirasi dari riset NeurIPS 2020) untuk menggantikan metode *Softmax/Entropy* standar. Sistem mengukur "stabilitas energi" dari sinyal *logits* mentah saraf AI. Objek asing (seperti gambar hewan atau makanan) akan memicu energi anomali yang tinggi dan otomatis ditolak oleh sistem.

### 4. 🚀 Auto-Active Learning & Dataset Manager
Sistem dirancang untuk terus bertumbuh:
- **Retrain Cepat (1 Epoch)**: Jika AI salah memprediksi telur saat *Deteksi Live*, Admin dapat langsung mengoreksi labelnya dan menekan tombol *Retrain*. AI akan langsung belajar menguasai pola tersebut dalam hitungan detik.
- **Dataset Manager**: Panel kontrol khusus untuk mengatur ribuan data gambar (*Train & Test*) yang dilengkapi dengan sistem *Pagination* dan penghapusan data spesifik untuk kemudahan navigasi.

### 5. 📊 Dashboard Analitik & Ekspor Laporan
Data adalah kunci bisnis. Sistem secara otomatis merekam setiap pemindaian ke dalam *database* (SQLite) lokal:
- Menampilkan grafik interaktif harian (menggunakan **Plotly**) mengenai jumlah telur segar dan busuk yang terdeteksi.
- Fitur **Download CSV** sekali klik untuk kebutuhan laporan harian maupun audit ke pihak manajemen.

---

## 🛠️ Arsitektur Teknologi (Tech Stack)

- **Frontend & Web Server**: [Streamlit](https://streamlit.io/) (Python)
- **Machine Learning & Computer Vision**: PyTorch, TorchVision, OpenCV
- **Pre-Trained Architectures**: MobileNetV2, ResNet18, SqueezeNet (Dengan kustomisasi *Deep Unfreezing* & *Color Jittering*)
- **Database**: SQLite3
- **Data Visualization**: Pandas, Plotly Express

---

## 🔬 Sorotan Metrik & Kinerja AI (Advanced ML)

Aplikasi ini tidak hanya menggunakan pemodelan standar, namun disematkan dengan teknik kompetisi tingkat lanjut:
1. **Test-Time Augmentation (TTA)**: Saat inference/memprediksi, gambar diputar dan dibalik (*flipped*) secara gaib di latar belakang sebanyak 4 kali, lalu probabilitasnya dirata-rata untuk memastikan stabilitas *confidence* hingga 99%.
2. **Kekebalan Cahaya (*Brightness Jittering*)**: Pelatihan model dimodifikasi secara matematis agar AI tidak kebingungan meskipun foto telur (*candling*) diambil menggunakan senter HP yang redup atau di ruangan yang terlalu terang.

---

*Didokumentasikan khusus untuk kebutuhan Portofolio Profesional & Presentasi Kompetisi.*
