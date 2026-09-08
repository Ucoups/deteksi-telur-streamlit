import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import json
import random
import datetime
from torch.utils.data import Dataset, DataLoader
from typing import Generator, Dict, Any

from utils.config import MODEL_PATH, MODEL_DIR, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, LEARNING_RATE
from utils.model_utils import build_model, get_device

class EggDataset(Dataset):
    """
    Dataset kustom untuk memuat data gambar telur berdasarkan
    daftar record yang tersimpan di database.
    """
    def __init__(self, records, data_dir, transform=None):
        self.records = records  # list of (filename, label)
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(list(set(label.upper() for _, label in records)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        filename, label = self.records[idx]
        img_path = os.path.join(self.data_dir, label.lower(), filename)
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label.upper()]

def train_model_stream(data_dir: str, model_name: str = "mobilenetv2", epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE, lr: float = LEARNING_RATE) -> Generator[Dict[str, Any], None, None]:
    """
    Fungsi generator yang akan melatih model PyTorch dan 
    me-yield progress ke Streamlit UI, lengkap dengan metrik validasi.
    """
    if not os.path.exists(data_dir):
        yield {"status": "error", "message": f"Folder {data_dir} tidak ditemukan!"}
        return

    # Sinkronisasi file manual terlebih dahulu dan ambil data dari DB
    try:
        from utils.database import sync_manual_files, get_connection
        yield {"status": "info", "message": "Mensinkronkan database dengan folder dataset..."}
        sync_manual_files()
        
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, label, dataset_split FROM dataset_training")
            db_records = cursor.fetchall()
    except Exception as e:
        yield {"status": "error", "message": f"Gagal memproses database: {str(e)}"}
        return

    # Filter gambar yang benar-benar ada di disk
    valid_train_records = []
    valid_test_records = []
    
    for filename, label, split in db_records:
        img_path = os.path.join(data_dir, label.lower(), filename)
        if os.path.exists(img_path):
            if split.upper() == 'TEST':
                valid_test_records.append((filename, label))
            else:
                valid_train_records.append((filename, label))

    if len(valid_train_records) == 0:
        yield {"status": "error", "message": "Tidak ada data training yang valid ditemukan di disk!"}
        return

    # Fallback jika data test kosong
    is_fallback_split = False
    if len(valid_test_records) == 0:
        if len(valid_train_records) >= 5:
            is_fallback_split = True
            random.seed(42)
            temp = list(valid_train_records)
            random.shuffle(temp)
            split_idx = int(len(temp) * 0.8)
            valid_train_records = temp[:split_idx]
            valid_test_records = temp[split_idx:]
        else:
            # Terlalu sedikit data, duplikasi train untuk test
            valid_test_records = list(valid_train_records)

    # Dapatkan daftar kelas unik yang diurutkan secara alfabetis
    all_records = valid_train_records + valid_test_records
    class_names = sorted(list(set(label.upper() for _, label in all_records)))
    
    if len(class_names) < 2:
        yield {"status": "error", "message": f"Minimal butuh 2 folder class (segar, busuk) untuk training. Ditemukan: {class_names}"}
        return
        
    device = get_device()
    msg = f"Ditemukan {len(valid_train_records)} data Train dan {len(valid_test_records)} data Test."
    if is_fallback_split:
        msg += " (Data Test kosong, membagi otomatis 20% dari Train sebagai Test)."
    yield {"status": "info", "message": msg}
    yield {"status": "info", "message": f"Kategori kelas: {class_names}"}
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        train_dataset = EggDataset(valid_train_records, data_dir, transform=train_transform)
        val_dataset = EggDataset(valid_test_records, data_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        yield {"status": "error", "message": f"Error load dataset: {str(e)}"}
        return
        
    yield {"status": "info", "message": f"Memuat arsitektur model {model_name}..."}
    
    model = build_model(model_name=model_name, num_classes=len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    
    yield {"status": "info", "message": "Memulai pelatihan model..."}
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    for epoch in range(epochs):
        # ─── Fase Training ───
        model.train()
        running_loss = 0.0
        running_corrects = 0
        batches_total = len(train_loader)
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            yield {
                "status": "progress_batch", 
                "epoch": epoch+1,
                "total_epochs": epochs,
                "batch": batch_idx+1,
                "total_batches": batches_total
            }
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (running_corrects.double() / len(train_dataset)).item() * 100
        
        # ─── Fase Validation (Test) ───
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = (val_corrects.double() / len(val_dataset)).item() * 100
        
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        yield {
            "status": "progress_epoch",
            "epoch": epoch+1,
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc
        }
        
    # ─── Menyimpan Model dan Metrik ───
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'model_name': model_name
    }, MODEL_PATH)

    
    # Menghitung Confusion Matrix di akhir
    num_classes = len(class_names)
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    # Format metrik akhir untuk disimpan ke JSON
    eval_metrics = {
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "confusion_matrix": confusion_matrix.tolist(),
        "class_names": class_names,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metrics_path = os.path.join(MODEL_DIR, 'evaluation_metrics.json')
    try:
        with open(metrics_path, 'w') as f:
            json.dump(eval_metrics, f, indent=4)
    except Exception as e:
        pass
        
    yield {
        "status": "done", 
        "message": f"Model berhasil disimpan di {MODEL_PATH}",
        "metrics": eval_metrics
    }


def train_quick_epoch(img_bytes: bytes, label: str) -> tuple[bool, str]:
    """
    Melakukan 'Active Learning' cepat (1 epoch, 1 image) untuk memperbarui bobot model.
    """
    import io
    from PIL import Image

    if not os.path.exists(MODEL_PATH):
        return False, "Model belum pernah dilatih sama sekali. Harap latih model utama (Full Train) terlebih dahulu!"

    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']
    model_name = checkpoint.get('model_name', 'mobilenetv2')

    label_upper = label.upper()
    if label_upper not in class_names:
        return False, f"Label '{label}' tidak ada di daftar class model: {class_names}. Latih ulang model penuh terlebih dahulu."

    class_idx = class_names.index(label_upper)

    model = build_model(model_name=model_name, num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        target_tensor = torch.tensor([class_idx], dtype=torch.long).to(device)

        model.train()
        criterion = nn.CrossEntropyLoss()
        # LR sangat kecil agar tidak merusak bobot yang sudah ada
        optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

        optimizer.zero_grad()
        outputs = model(input_tensor)
        loss = criterion(outputs, target_tensor)
        loss.backward()
        optimizer.step()

        # Simpan kembali
        checkpoint['model_state_dict'] = model.state_dict()
        torch.save(checkpoint, MODEL_PATH)

        # Hapus cache model lama di ml_inference agar model baru dimuat
        import utils.ml_inference as mlinf
        mlinf.model_cache = None

        return True, "Model berhasil menyerap gambar ini! Bobot telah diperbarui."
    except Exception as e:
        return False, f"Terjadi kesalahan saat melatih: {str(e)}"


