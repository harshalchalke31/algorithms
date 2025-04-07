from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F
from gaf import GAF

class ECGDataset(Dataset):
    def __init__(self, signals, labels,gaf_mode:str='both', pad:bool=False):
        self.signals = signals
        self.labels = labels
        self.gaf_mode=gaf_mode
        self.pad = pad

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]

        if self.gaf_mode=='both':

            # Compute GAF (summation & difference)
            gaf_sum = GAF(signal, summation=True)
            gaf_diff = GAF(signal, summation=False)

            # Stack channels and convert to tensor
            gaf_img = np.stack([gaf_sum, gaf_diff], axis=0)  # (2, 187, 187)
            gaf_img = torch.tensor(gaf_img, dtype=torch.float32)

        elif self.gaf_mode=='gasf':
            gaf_sum = GAF(signal, summation=True)
            gaf_img = torch.tensor(gaf_sum, dtype=torch.float32).unsqueeze(0)

        elif self.gaf_mode=='gadf':
            gaf_diff = GAF(signal, summation=False)
            gaf_img = torch.tensor(gaf_diff, dtype=torch.float32).unsqueeze(0)

        if self.pad:
            # Adaptive padding to (2, 224, 224)
            pad = (18, 19, 18, 19)  # (left, right, top, bottom)
            gaf_img = F.pad(gaf_img, pad=pad)  # (2, 224, 224)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return gaf_img, label



def plot_classifier_curves(log_path: Path, suptitle: str = 'Classifier Performance'):
    df = pd.read_csv(log_path)

    epochs = df['Epoch']
    train_loss = df['Train Loss']
    val_loss = df['Val Loss']
    train_acc = df['Train Acc']
    val_acc = df['Val Acc']

    plt.figure(figsize=(12, 5))

    # --- Subplot 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # --- Subplot 2: Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_gaf_samples(dataset, num_samples=6, cols=3, gaf_type="GASF"):
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(num_samples):
        gaf_img, label = dataset[i]
        gaf_img = gaf_img.squeeze(0).numpy()  # shape: (187, 187)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(gaf_img, cmap='rainbow', origin='upper')
        plt.title(f"{gaf_type} - Label: {label.item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_gaf_both(dataset, num_samples=6, cols=3, gaf_type="GASF + GADF"):
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows * 0.5))  # reduce height since 2 plots per sample

    for i in range(num_samples):
        gaf_img, label = dataset[i]  # shape: (2, 187, 187)
        
        # Plot GASF
        plt.subplot(rows * 2, cols, i * 2 + 1)
        plt.imshow(gaf_img[0].numpy(), cmap='rainbow', origin='upper')
        plt.title(f"GASF - Label: {label.item()}")
        plt.axis('off')

        # Plot GADF
        plt.subplot(rows * 2, cols, i * 2 + 2)
        plt.imshow(gaf_img[1].numpy(), cmap='rainbow', origin='upper')
        plt.title(f"GADF - Label: {label.item()}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def evaluate_classifier(model, dataloader, device, class_names=None, title="Confusion Matrix"):
    model.eval().to(device)
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Print metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()
