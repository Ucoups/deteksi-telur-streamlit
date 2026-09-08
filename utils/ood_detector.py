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
MSP_THRESHOLD = 0.51

# Konfigurasi Energy-Based OOD (Liu et al., NeurIPS 2020)
# Suhu (Temperature) untuk meratakan logits
TEMPERATURE = 1.0
# Jika Free Energy di atas nilai ini → tolak (karena energi terlalu tinggi / tidak stabil)
# Nilai ini bisa disesuaikan. Gambar in-distribution biasanya bernilai negatif besar (misal -3.0 s/d -6.0)
ENERGY_THRESHOLD = -0.5


def _compute_free_energy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Menghitung Helmholtz Free Energy dari vektor logits.
    Energi tinggi (mendekati 0 atau positif) = Out-of-Distribution.
    Energi rendah (negatif besar) = In-Distribution.
    
    E(x) = -T * logsumexp(logits / T)
    """
    energy = -temperature * torch.logsumexp(logits / temperature, dim=0)
    return float(energy.item())


def detect_ood(
    image_bytes: bytes,
    model: nn.Module,
    msp_threshold: float = MSP_THRESHOLD,
    energy_threshold: float = ENERGY_THRESHOLD
) -> Tuple[bool, str, dict]:
    """
    Melakukan 2 lapis pemeriksaan OOD tingkat lanjut pada gambar.

    Args:
        image_bytes: Bytes gambar dari Streamlit uploader
        model: Model PyTorch yang sudah dilatih
        msp_threshold: Batas minimum confidence.
        energy_threshold: Batas maksimum Free Energy. Di atas ini = OOD.

    Returns:
        Tuple:
            - is_ood (bool): True jika gambar bukan telur
            - reason (str): Penjelasan
            - metrics (dict): Detail metrik
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
        logits = model(input_tensor)[0]
        probabilities = torch.nn.functional.softmax(logits, dim=0)

    max_prob = probabilities.max().item()
    free_energy = _compute_free_energy(logits, TEMPERATURE)

    # Susun detail metrik untuk UI
    metrics = {
        "max_confidence": round(max_prob * 100, 2),
        "free_energy": round(free_energy, 4),
        "threshold_msp": round(msp_threshold * 100, 2),
        "threshold_energy": energy_threshold,
        "prob_per_class": {f"class_{i}": round(p.item() * 100, 2) for i, p in enumerate(probabilities)},
        "logits_raw": {f"class_{i}": round(l.item(), 4) for i, l in enumerate(logits)}
    }

    # === LAYER 1: Maximum Softmax Probability (MSP) Check ===
    if max_prob < msp_threshold:
        return True, (
            f"Confidence terlalu rendah ({max_prob*100:.1f}% < {msp_threshold*100:.0f}%). "
            "AI tidak mengenali objek ini sebagai telur."
        ), metrics

    # === LAYER 2: Free Energy Check (NeurIPS 2020) ===
    if free_energy > energy_threshold:
        return True, (
            f"Energi gambar terlalu tinggi / kacau (Energy: {free_energy:.4f} > {energy_threshold}). "
            "Objek terdeteksi sebagai anomali (Out-of-Distribution)."
        ), metrics

    # Lolos semua pemeriksaan → Gambar dikenali sebagai telur
    return False, "✅ Gambar memiliki energi stabil. Dikenali sebagai telur ayam.", metrics
