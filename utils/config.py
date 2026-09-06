import os

# Database Paths
DB_PATH = 'database/produksi_telur.db'
IMAGE_DIR = 'database/dataset_images'

# Model Paths
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'egg_classifier.pkl')

# ML Hyperparameters Default
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
LEARNING_RATE = 0.001

# Labels
CLASS_LABELS = ["SEGAR", "BUSUK"]

# User Accounts Configuration (untuk sistem login & pembatasan hak akses)
USERS = {
    "admin": {
        "password": "admin123",
        "role": "admin",
        "name": "Administrator Sistem"
    },
    "operator": {
        "password": "user123",
        "role": "user",
        "name": "Staf Operator Telur"
    }
}

