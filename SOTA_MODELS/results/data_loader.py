# data_loader.py
import os
import cv2
import gc
import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader

# ==========================================
# CUSTOM MEMORY-EFFICIENT DATASET
# ==========================================
class MemoryEfficientDataset(Dataset):
    """
    Keeps images in RAM as lightweight uint8 (0-255) to save memory.
    Converts to heavy float32 (0.0-1.0) ON THE FLY only when a batch is requested.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 1. Get the uint8 image
        img = self.X[idx]

        # 2. Convert to float32 and scale to [0, 1] ONLY for this specific image
        # Using ascontiguousarray prevents stride warnings from PyTorch
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0

        # 3. Get label
        label_tensor = torch.tensor(self.y[idx], dtype=torch.long)

        return img_tensor, label_tensor

# ==========================================
# WORKER FUNCTION
# ==========================================
def _process_image_worker(file_path, target_size, label):
    """Worker function to read, resize, convert to RGB, and extract grouping info."""
    try:
        # Read image
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None

        # Convert BGR (OpenCV default) to RGB (PyTorch standard)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img_resized = cv2.resize(img, target_size)
        filename = os.path.basename(file_path)

        return (img_resized, label, filename)
    except Exception:
        return None

# ==========================================
# MAIN DATA SPLIT FUNCTION
# ==========================================
def get_leakage_free_split(dataset_path, target_size, test_size=0.2, random_state=42, batch_size=32):
    """
    Loads images, prevents augmentation leakage, and returns PyTorch DataLoaders.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"Classes found: {class_names}")

    tasks = []
    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(dataset_path, class_name)
        files = os.listdir(class_folder)
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        for f in files:
            if f.lower().endswith(valid_exts):
                tasks.append((os.path.join(class_folder, f), idx))

    print(f"Total images found: {len(tasks)}")
    print("Starting Parallel Image Loading...")

    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(_process_image_worker)(path, target_size, lbl)
        for path, lbl in tqdm(tasks)
    )

    results = [r for r in results if r is not None]

    # Keep as lightweight uint8
    X_img = np.array([r[0] for r in results], dtype=np.uint8)
    y_raw = np.array([r[1] for r in results], dtype=np.int64)
    filenames = [r[2] for r in results]

    del results
    gc.collect()

    # Create Group IDs for leakage prevention
    groups = []
    for fname in filenames:
        if fname.startswith("aug_"):
            parts = fname.split("_")
            original_name = "_".join(parts[2:]) if len(parts) > 2 else fname
            groups.append(original_name)
        else:
            groups.append(fname)

    groups = np.array(groups)

    print(f"\nSplitting data using GroupShuffleSplit (Preventing Data Leakage)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X_img, y_raw, groups=groups))

    X_train, X_val = X_img[train_idx], X_img[val_idx]
    y_train, y_val = y_raw[train_idx], y_raw[val_idx]

    del X_img, y_raw, groups
    gc.collect()

    print(f"Successfully Split -> Train Set: {len(X_train)} | Val Set: {len(X_val)}")

    # ==========================================
    # PYTORCH CONVERSIONS
    # ==========================================
    print("Preparing Memory-Efficient PyTorch DataLoaders...")

    # 1. PyTorch expects NCHW format instead of NHWC
    # We transpose here, but keep them as uint8 numpy arrays!
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_val = np.transpose(X_val, (0, 3, 1, 2))

    # 2. Wrap in our Custom Memory Efficient Dataset
    train_dataset = MemoryEfficientDataset(X_train, y_train)
    val_dataset = MemoryEfficientDataset(X_val, y_val)

    # 3. Create DataLoaders
    # num_workers=0 is safest for Windows locally. If on Linux/Colab, you can increase it.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, class_names
