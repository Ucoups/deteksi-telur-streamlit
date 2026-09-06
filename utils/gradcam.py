"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
=======================================================
Modul ini menghasilkan peta panas (heatmap) yang menunjukkan
bagian gambar MANA yang paling diperhatikan AI saat membuat keputusan.

Referensi: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
from typing import Optional, Tuple
from torchvision import transforms


class GradCAM:
    """
    Implementasi Grad-CAM untuk MobileNetV2.
    Grad-CAM bekerja dengan:
    1. Menjalankan forward pass
    2. Menghitung gradient dari output class terhadap feature map layer terakhir
    3. Global Average Pooling pada gradient → bobot per channel
    4. Mengalikan bobot dengan activation map → heatmap
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        """Mendaftarkan forward & backward hooks pada layer konvolusi terakhir secara dinamis."""
        target_layer = None
        # Cari layer Conv2d paling terakhir dalam model
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
                
        if target_layer is None:
            # Fallback ke features[-1] jika tidak ditemukan
            if hasattr(self.model, 'features') and len(self.model.features) > 0:
                target_layer = self.model.features[-1]
            else:
                target_layer = list(self.model.children())[-2]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)


    def generate_heatmap(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Menghasilkan heatmap Grad-CAM (nilai 0.0 - 1.0).

        Args:
            input_tensor: Tensor gambar yang sudah di-transform (shape: [1, C, H, W])
            class_idx: Index class target. Jika None, pakai class dengan skor tertinggi.

        Returns:
            np.ndarray heatmap ternormalisasi (H x W), float32 [0, 1]
        """
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass untuk class tertentu
        score = output[0, class_idx]
        score.backward()

        # Hitung bobot: rata-rata gradients di setiap channel
        # Shape gradients: [1, C, H, W]
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Hitung weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)  # Hanya nilai positif yang relevan

        # Normalisasi ke [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def apply_gradcam_overlay(
    original_image_bytes: bytes,
    model: nn.Module,
    class_idx: Optional[int] = None,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fungsi utama: menerima bytes gambar, menghasilkan gambar overlay Grad-CAM.

    Returns:
        Tuple of:
            - overlay_image: Gambar original + heatmap (RGB numpy array)
            - heatmap_only: Heatmap dalam bentuk warna jet (RGB numpy array)
    """
    # Transformasi gambar
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_pil = Image.open(io.BytesIO(original_image_bytes)).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0)

    # Generate heatmap
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_heatmap(input_tensor, class_idx)

    # Resize heatmap ke ukuran gambar asli (224x224 setelah crop)
    img_resized = img_pil.resize((224, 224))
    img_np = np.array(img_resized, dtype=np.uint8)

    # Konversi heatmap ke warna (colormap MAGMA = premium dan mudah dibaca juri)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (224, 224))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Buat overlay dengan blending
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_rgb, alpha, 0)

    return overlay, heatmap_rgb
