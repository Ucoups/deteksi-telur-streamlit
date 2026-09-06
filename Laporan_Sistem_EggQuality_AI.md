# LAPORAN TEKNIS KOMPREHENSIF: MODEL KECERDASAN BUATAN (AI) DETEKSI KUALITAS TELUR

Dokumen ini mendeskripsikan secara mendalam arsitektur, landasan matematis, proses pelatihan, optimasi, dan metode evaluasi dari model *Machine Learning* yang digunakan dalam sistem deteksi kualitas telur.

---

## 1. LANDASAN ARSITEKTUR MODEL (MODEL ARCHITECTURE)

Sistem ini didayagai oleh algoritma **Convolutional Neural Network (CNN)** menggunakan framework **PyTorch**. CNN dipilih karena kemampuannya dalam mengekstraksi fitur hierarkis spasial dari gambar (mulai dari deteksi tepi cangkang hingga tekstur spesifik seperti bercak kotoran atau retakan).

### A. Pendekatan Transfer Learning
Daripada melatih model dari bobot acak (*random initialization*) yang membutuhkan ribuan dataset dan GPU berkinerja tinggi, sistem ini menggunakan teknik **Transfer Learning** dari model yang telah dilatih (*pre-trained*) menggunakan dataset raksasa ImageNet (1.2 juta gambar, 1000 kelas).

### B. Arsitektur yang Didukung
Sistem mengintegrasikan tiga jenis arsitektur tulang punggung (*backbone*) yang dapat disesuaikan dengan kebutuhan komputasi:
1.  **MobileNetV2 (Default):** Menggunakan arsitektur *"Inverted Residuals and Linear Bottlenecks"*. Sangat efisien karena menggunakan *Depthwise Separable Convolutions*, secara drastis mengurangi jumlah parameter matematis tanpa mengorbankan akurasi secara signifikan. Sangat ideal untuk eksekusi berbasis CPU murni.
2.  **ResNet-18 (Residual Network):** Memecahkan masalah *"Vanishing Gradient"* pada jaringan saraf yang dalam dengan menggunakan *"Skip Connections"* atau jalan pintas. Arsitektur ini sangat stabil dan direkomendasikan jika corak perbedaan antara telur segar dan busuk sangat halus/samar.
3.  **SqueezeNet:** Memanfaatkan *"Fire Modules"* yang terdiri dari lapisan *squeeze* (filter 1x1) dan lapisan *expand* (filter 1x1 dan 3x3). Ini menghasilkan model dengan ukuran file sangat kecil (kurang dari 5MB), sangat cocok untuk sistem dengan keterbatasan memori penyimpanan.

### C. Modifikasi Lapisan (Freeze & Replace)
Pada tingkat kode, sistem membekukan (*freezing*) seluruh bobot fitur ekstraksi dasar dengan perintah `param.requires_grad = False`. Sistem hanya mengisolasi dan memotong lapisan klasifikasi terakhir (misal: `model.fc` pada ResNet atau `model.classifier[1]` pada MobileNet/SqueezeNet). Lapisan ini kemudian diganti dengan **Fully Connected Layer** baru yang *output*-nya dipetakan secara eksklusif ke jumlah kelas target dataset kita (Misal: 2 node untuk SEGAR dan BUSUK). Hanya matriks bobot pada lapisan akhir inilah yang akan diperbarui (dilatih) selama proses *backpropagation*.

---

## 2. PRA-PEMROSESAN & AUGMENTASI DATA (DATA PIPELINE)

Input dari kamera atau file tidak bisa langsung dimasukkan ke jaringan saraf. Sistem memiliki tahapan *Data Pipeline* (menggunakan modul `torchvision.transforms`) yang ketat:

### A. Pra-pemrosesan Standar (Train & Inference)
*   **Resize & Center Crop:** Gambar diubah ukurannya ke dimensi 256x256 piksel, lalu dipotong secara presisi di tengah menjadi **224x224 piksel**. Dimensi 224x224 adalah standar matematis wajib (*input tensor geometry*) untuk arsitektur *ImageNet pre-trained*.
*   **Tensor Conversion (`ToTensor`):** Mengonversi matriks piksel gambar berformat rentang asli `[0, 255]` menjadi format *Tensor* (matriks matematis PyTorch) dengan rentang `[0.0, 1.0]`.
*   **ImageNet Normalization:** Memusatkan dan menstandarkan distribusi warna menggunakan mean (rata-rata) `[0.485, 0.456, 0.406]` dan standard deviation (simpangan baku) `[0.229, 0.224, 0.225]` untuk masing-masing saluran warna Red, Green, Blue (RGB). Hal ini mutlak diperlukan agar nilai intensitas gambar sejalan dengan rentang nilai yang digunakan saat model awal dilatih.

### B. Augmentasi Data (Khusus Training)
Untuk mencegah model menghafal dataset secara statis (*Overfitting*), sistem menerapkan mutasi gambar acak secara real-time saat *training*:
*   **Random Horizontal & Vertical Flip:** Probabilitas 50% membalik sumbu X dan Y dari gambar telur.
*   **Random Rotation:** Memutar posisi gambar telur secara acak dalam rentang sudut -15 derajat hingga +15 derajat.

Kombinasi augmentasi ini memastikan bahwa meskipun dataset fisik jumlahnya terbatas, variasi matriks yang dilihat oleh model di setiap putaran akan selalu berbeda. Hal ini membuat model sangat *"Robust"* terhadap posisi letak telur yang tidak beraturan.

---

## 3. PROSES PELATIHAN DAN OPTIMASI MATEMATIS

Fase pelatihan dikoordinasikan oleh fungsi *generator* yang secara efisien memproses memori dalam potongan kecil (tidak memuat seluruh gambar ke RAM PC sekaligus).

*   **Pembagian Proporsional (Train/Test Split):** Menerapkan prinsip pareto 80/20. 80% data didedikasikan untuk pencarian bobot gradien (*training*), sementara 20% data disembunyikan secara total dari model untuk pengujian objektif (*validation*) di akhir setiap putaran.
*   **Adam Optimizer (Adaptive Moment Estimation):** Menggantikan Stochastic Gradient Descent (SGD) klasik. Adam menghitung kecepatan belajar (*learning rate*) yang adaptif untuk setiap parameter individu menggunakan estimasi momen orde pertama dan kedua dari gradien, membuat konvergensi (pencapaian akurasi maksimal) tercapai jauh lebih mulus dan cepat.
*   **Fungsi Kerugian (Cross-Entropy Loss):** Digunakan karena ini adalah fungsi logaritmik yang sangat "menghukum" model (memberikan nilai error tinggi) jika model sangat yakin pada tebakan yang ternyata salah. Rumus matematisnya memaksa penyebaran probabilitas kelas mendekati nilai kebenaran murni (*Ground Truth*).
*   **Iterasi Batch & Epoch:** Gambar disuplai ke model dalam bentuk ukuran *Batch* (misal: 4 gambar sekaligus). Satu *Epoch* terhitung selesai jika model telah mengevaluasi dan memperbarui bobotnya menggunakan keseluruhan total gambar pada dataset latih.

---

## 4. MEKANISME INFERENSI (DETEKSI WAKTU NYATA)

Saat digunakan di antarmuka sistem deteksi, alur eksekusinya dioptimalkan secara khusus untuk menjamin eksekusi yang cepat (*Low Latency*):

*   **Caching Model In-Memory:** Model di-load dari media penyimpanan (`.pth`) ke dalam RAM hanya pada pemicuan deteksi pertama (`model_cache`). Deteksi telur selanjutnya akan mengeksekusi arsitektur secara langsung dari memori tanpa melalui jeda baca/tulis disk.
*   **Mode Evaluasi & No-Gradient:** Arsitektur dikunci secara paksa pada mode statis (`model.eval()`) dan komputasi dibungkus dalam blok konteks `torch.no_grad()`. Hal ini mematikan pelacakan kalkulus gradien (algoritma *autograd*) sehingga menekan alokasi penggunaan RAM dan membebaskan siklus kerja CPU secara signifikan (mempercepat prediksi hingga 3x lipat).
*   **Injeksi Dimensi Batch:** Karena jaringan CNN selalu berekspektasi masukan berupa format 4-Dimensi `[Batch, Channel, Height, Width]`, gambar tunggal secara teknis dimanipulasi dengan instruksi `unsqueeze(0)` untuk menyuntikkan dimensi *dummy* (batch bernilai 1) di depan matriks gambar sebelum diumpankan ke model.
*   **Probabilitas Softmax:** Skor mentah (*Logits*) berbentuk distribusi linear dari ujung lapisan output model tidak bisa langsung dibaca manusia. Skor tersebut dikonversi menggunakan rumusan fungsi aktivasi **Softmax** ke dalam rentang eksponensial nilai 0 hingga 1.0 (merepresentasikan rentang kepercayaan probabilitas 0% hingga 100%).

---

## 5. EVALUASI DAN VALIDASI PERFORMA METRIK

Tingkat kepintaran dan reliabilitas algoritma dievaluasi secara ketat dan saintifik pada ujung fase pelatihan:

### A. Analisis Kurva Real-time
*   **Training vs Validation Loss:** Kurva *loss* merepresentasikan tingkat "kesalahan" prediksi model. Indikator pelatihan sukses dan sehat adalah jika garis *Training Loss* menurun seiring dan berhimpitan dengan *Validation Loss*. Jika *Training Loss* menukik tajam ke bawah sementara *Validation Loss* justru melengkung naik, fenomena tersebut menandakan *Overfitting* (model hanya sukses pada data latih, tapi gagal pada tantangan dunia nyata).
*   **Training vs Validation Accuracy:** Kurva ini merepresentasikan laju pertumbuhan tebakan benar dari total eksperimen komputasi di setiap putaran *Epoch*.

### B. Matriks Kebingungan (Confusion Matrix)
Merupakan tabel pembedahan matriks dua dimensi yang menelanjangi sifat tebakan akhir dari arsitektur:
*   **True Positive (TP):** Aktual Busuk, Prediksi Busuk. *(Kinerja Sempurna)*.
*   **True Negative (TN):** Aktual Segar, Prediksi Segar. *(Kinerja Sempurna)*.
*   **False Positive (FP / Type I Error):** Aktual Segar, Prediksi Busuk. *(Alarm Palsu. Sebuah telur bagus tetapi kecerdasan buatan menyuruh membuangnya, merugikan peternak secara material)*.
*   **False Negative (FN / Type II Error):** Aktual Busuk, Prediksi Segar. *(Kesalahan Paling Fatal. Telur busuk berpenyakit lolos seleksi mesin dan terdistribusi hingga piring konsumen. Kesalahan ini (FN) harus ditekan sekecil mungkin pada model)*.

Dengan memecah formula akurasi dasar `(TP + TN) / (TP + TN + FP + FN)` ke dalam tabel kuadran Confusion Matrix, peneliti dan pihak administrasi sistem dapat melakukan investigasi spesifik letak titik kelemahan model secara presisi, alih-alih sekadar membaca agregat persentase akurasi buta semata.
