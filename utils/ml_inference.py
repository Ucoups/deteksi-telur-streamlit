import torch
from torchvision import transforms
from PIL import Image
import os
import io
from typing import Tuple, Optional, List
from utils.config import MODEL_PATH
from utils.model_utils import build_model, get_device

# Cache model agar tidak perlu dimuat ulang setiap klik
model_cache = None
class_names_cache = None

def load_model() -> Tuple[Optional[torch.nn.Module], Optional[List[str]]]:
    """Memuat model dari disk dan menyimpannya di cache memory."""
    global model_cache, class_names_cache
    if model_cache is not None:
        return model_cache, class_names_cache
        
    if not os.path.exists(MODEL_PATH):
        return None, None
        
    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']
    model_name = checkpoint.get('model_name', 'mobilenetv2')
    
    # Membangun kembali arsitektur
    model = build_model(model_name=model_name, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()
    
    model_cache = model
    class_names_cache = class_names
    return model, class_names

def predict_egg(image_bytes: bytes) -> Tuple[str, float]:
    """
    Fungsi menerima file gambar bytes dari Streamlit dan mengembalikan hasil prediksi sesungguhnya.
    """
    model, class_names = load_model()
    
    if model is None or class_names is None:
        return "MODEL_BELUM_DILATIH", 0.0
        
    device = get_device()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    t_base = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    t_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 4 Test-Time Augmentation (TTA) Passes
    import torchvision.transforms.functional as TF
    tta_funcs = [
        lambda x: x,                     # Original
        TF.hflip,                        # Horizontal
        TF.vflip,                        # Vertical
        lambda x: TF.rotate(x, 90)       # Rotasi 90 derajat
    ]
    
    base_img = t_base(img)
    probs_list = []
    
    with torch.no_grad():
        for t_func in tta_funcs:
            aug_img = t_func(base_img)
            input_tensor = t_norm(aug_img).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            prob = torch.nn.functional.softmax(outputs[0], dim=0)
            probs_list.append(prob)
            
        # Rata-rata dari 4 prediksi
        avg_probs = torch.stack(probs_list).mean(dim=0)
        confidence, predicted_idx = torch.max(avg_probs, 0)
        
    label = class_names[predicted_idx.item()].upper()
    return label, confidence.item()
