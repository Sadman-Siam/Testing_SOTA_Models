import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.transforms import Normalize
from sklearn.metrics import classification_report, confusion_matrix

# IMPORT OUR CUSTOM LEAKAGE-FREE LOADER (PyTorch Version)
from data_loader import get_leakage_free_split

# =========================================================
# 1. Configuration
# =========================================================
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 200
DATA_DIR = "../augmented_balanced_2000_256"
RESULTS_DIR = "./results"
RESULT_PREFIX = "mobilenet_v2"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Setup Device (Use GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Compute Device: {device}")

# Pre-trained models expect this specific ImageNet normalization
imagenet_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# =========================================================
# 2. Main Execution
# =========================================================
def main():
    print("=" * 60)
    print("Baseline SOTA Test: MobileNetV2 (PyTorch Leakage-Free)")
    print("=" * 60)

    # 1. Load Data using PyTorch DataLoaders
    train_loader, val_loader, classes = get_leakage_free_split(
        dataset_path=DATA_DIR,
        target_size=IMAGE_SIZE,
        test_size=0.2,
        batch_size=BATCH_SIZE
    )
    num_classes = len(classes)

    # 2. Build Absolutely Default MobileNetV2 Model
    print("\nBuilding Default MobileNetV2...")

    # Load native PyTorch MobileNetV2 with ImageNet weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Standard practice: freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer (classification head)
    # MobileNetV2 stores its head in `model.classifier[1]`
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)

    # Define Loss and Optimizer (Only optimize the newly added classifier layer)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=1e-3)

    # 3. Training Loop Setup (Emulating Keras callbacks/history)
    history = {'accuracy':[], 'val_accuracy': [], 'loss':[], 'val_loss':[]}
    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 10
    best_model_path = os.path.join(RESULTS_DIR, f"{RESULT_PREFIX}_best.pth")

    print("\nStarting Training...")

    for epoch in range(EPOCHS):
        # --- TRAIN PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = imagenet_normalize(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            train_total += labels.size(0)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = imagenet_normalize(inputs)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        # --- EPOCH CALCULATIONS ---
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history['loss'].append(epoch_train_loss)
        history['accuracy'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_acc)

        print(f"Epoch {epoch+1:03d}/{EPOCHS:03d} - "
              f"loss: {epoch_train_loss:.4f} - accuracy: {epoch_train_acc:.4f} - "
              f"val_loss: {epoch_val_loss:.4f} - val_accuracy: {epoch_val_acc:.4f}")

        # --- EARLY STOPPING & CHECKPOINT ---
        if epoch_val_acc > best_val_acc:
            print(f"   *** Val Acc improved from {best_val_acc:.4f} to {epoch_val_acc:.4f}. Saving model! ***")
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}. Restoring best weights.")
            break
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Evaluate (Using best weights)
    # ==========================================
    print("\nEvaluating Model...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_pred_classes =[]
    y_true_classes =[]

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            inputs = imagenet_normalize(inputs)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_pred_classes.extend(preds.cpu().numpy())
            y_true_classes.extend(labels.cpu().numpy())

    y_pred_classes = np.array(y_pred_classes)
    y_true_classes = np.array(y_true_classes)

    report = classification_report(y_true_classes, y_pred_classes, target_names=classes)
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(report)

    with open(f"{RESULTS_DIR}/{RESULT_PREFIX}_report.txt", "w") as f:
        f.write(report)

    # ==========================================
    # 5. Plot Accuracy & Loss
    # ==========================================
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history['val_accuracy'], label='Val Accuracy', color='orange', linewidth=2)
    plt.title('MobileNetV2 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    plt.title('MobileNetV2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{RESULT_PREFIX}_history.png")
    print(f"Saved training history plot to {RESULTS_DIR}")

    # ==========================================
    # 6. Plot Confusion Matrix
    # ==========================================
    cm = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('MobileNetV2 Confusion Matrix (Normalized)', fontsize=16, pad=15)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    plt.savefig(f"{RESULTS_DIR}/{RESULT_PREFIX}_cm.png", dpi=300)
    print(f"Saved confusion matrix plot to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
