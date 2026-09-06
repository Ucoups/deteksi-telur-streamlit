import torch
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
from utils.ml_inference import load_model
from utils.config import MODEL_DIR

def export_for_mobile():
    print("Mengekspor model PyTorch ke format Mobile (TorchScript)...")
    
    # 1. Muat model yang ada
    model, class_names = load_model()
    if model is None:
        print("Error: Model belum dilatih! Silakan latih model di Streamlit terlebih dahulu.")
        return
        
    model.eval()
    
    # 2. Buat tensor dummy dengan bentuk yang persis seperti gambar input (1 batch, 3 warna, 224x224 piksel)
    dummy_input = torch.rand(1, 3, 224, 224)
    
    # 3. Trace model (Merekam jejak komputasi model)
    try:
        traced_script_module = torch.jit.trace(model, dummy_input)
        
        # 4. Optimasi untuk Mobile (memangkas bagian yang tidak perlu di HP)
        optimized_mobile_model = optimize_for_mobile(traced_script_module)
        
        # 5. Simpan sebagai .ptl
        ptl_path = os.path.join(MODEL_DIR, "egg_classifier.ptl")
        optimized_mobile_model._save_for_lite_interpreter(ptl_path)
        
        # Simpan juga label kelasnya ke txt agar aplikasi Android tahu apa arti outputnya
        classes_path = os.path.join(MODEL_DIR, "classes.txt")
        with open(classes_path, "w") as f:
            for cls_name in class_names:
                f.write(f"{cls_name}\n")
                
        print(f"Sukses! Model mobile disimpan di: {ptl_path}")
        print(f"File label disimpan di: {classes_path}")
        print("File ini siap disalin ke dalam folder assets aplikasi Android (App/app/src/main/assets/).")
        
    except Exception as e:
        print(f"Gagal mengekspor model: {e}")

if __name__ == "__main__":
    export_for_mobile()
