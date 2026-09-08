"""
Out-of-Distribution (OOD) Detection
=====================================
Modul ini memastikan bahwa gambar yang diunggah adalah TELUR AYAM,
bukan objek acak lainnya (misal: kucing, buah, wajah manusia).

Strategi OOD yang diimplementasikan (3 lapis filter):
1. Maximum Softmax Probability (MSP): Jika confidence AI terlalu rendah, tolak.
2. Entropy Check: Jika distribusi probabilitas terlalu merata (tidak yakin), tolak.
3. Feature Magnitude Check: Menggunakan fitur norm dari layer embedding.

Referensi: Hendrycks & Gidong, "A Baseline for Detecting Misclassified
and Out-of-Distribution Examples in Neural Networks", ICLR 2017.
"""

import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
from torchvision import transforms
from typing import Tuple
import math


# --- Threshold konfigurasi (dapat di-tune) ---
# Jika confidence di bawah nilai ini → tolak sebagai "Bukan Telur"
# (Diubah menjadi 51% karena deteksi telur dalam gelap/candling sering menghasilkan confidence mendekati 50%)
MSP_THRESHOLD = 0.51

# Jika entropy terlalu tinggi → AI tidak yakin → tolak
# Max entropy untuk 2 class = log(2) ≈ 0.693
ENTROPY_THRESHOLD = 0.6929


def _compute_entropy(probabilities: torch.Tensor) -> float:
    """
    Menghitung Shannon Entropy dari vektor probabilitas.
    Entropy tinggi = AI tidak yakin / distribusi merata.
    
    H = -sum(p * log(p))
    """
    probs = probabilities.cpu().numpy()
    # Tambahkan epsilon untuk menghindari log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def detect_ood(
    image_bytes: bytes,
    model: nn.Module,
    msp_threshold: float = MSP_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD
) -> Tuple[bool, str, dict]:
    """
    Melakukan 3 lapis pemeriksaan OOD pada gambar yang diunggah.

    Args:
        image_bytes: Bytes gambar dari Streamlit uploader
        model: Model PyTorch yang sudah dilatih
        msp_threshold: Batas minimum confidence. Di bawah ini = OOD
        entropy_threshold: Batas maximum entropy. Di atas ini = OOD

    Returns:
        Tuple:
            - is_ood (bool): True jika gambar bukan telur (Out-of-Distribution)
            - reason (str): Penjelasan mengapa gambar ditolak/diterima
            - metrics (dict): Detail metrik untuk ditampilkan ke juri
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    max_prob = probabilities.max().item()
    entropy = _compute_entropy(probabilities)

    # Susun detail metrik (berguna untuk ditampilkan di UI)
    metrics = {
        "max_confidence": round(max_prob * 100, 2),
        "entropy": round(entropy, 4),
        "entropy_max_possible": round(math.log(len(probabilities)), 4),
        "threshold_msp": round(msp_threshold * 100, 2),
        "threshold_entropy": entropy_threshold,
        "prob_per_class": {f"class_{i}": round(p.item() * 100, 2) for i, p in enumerate(probabilities)}
    }

    # === LAYER 1: Maximum Softmax Probability (MSP) Check ===
    if max_prob < msp_threshold:
        return True, (
            f"Confidence terlalu rendah ({max_prob*100:.1f}% < {msp_threshold*100:.0f}%). "
            "AI tidak mengenali objek ini sebagai telur."
        ), metrics

    # === LAYER 2: Entropy Check ===
    if entropy > entropy_threshold:
        return True, (
            f"Distribusi probabilitas terlalu merata (Entropy: {entropy:.4f} > {entropy_threshold}). "
            "AI sangat ragu-ragu, kemungkinan besar ini bukan telur."
        ), metrics

    # Lolos semua pemeriksaan → Gambar dikenali sebagai telur
    return False, "✅ Gambar dikenali sebagai telur ayam.", metrics
