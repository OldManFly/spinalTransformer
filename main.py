import torch
from vit import ViT
from dataLoader import spinalLoader
from torch.utils.data import DataLoader
from torch import nn
import os
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score  # 你可能不會直接用 accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from utls import dice_loss  # 確保你導入了 dice_loss

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # 確保 labels 是 long 類型，並且 outputs 是 logits
        labels = labels.long()
        # print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")

        if criterion.__class__.__name__ == 'CrossEntropyLoss':
            loss = criterion(outputs, labels.squeeze(1))  # 移除 labels 的通道維度 (如果存在)
        elif criterion.__name__ == 'dice_loss':
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # 追蹤分割結果 (你需要根據你的具體任務調整)
        preds = outputs.argmax(dim=1).cpu().numpy()  # 取得每個像素的預測類別
        true_labels = labels.cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(true_labels.tolist())

    avg_loss = total_loss / len(train_loader.dataset)
    # 你可能需要計算 Dice 係數或其他分割指標，而不是準確度
    # acc = accuracy_score(all_labels, all_preds)
    # print(f"Training Accuracy: {acc}")

    return avg_loss

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            labels = labels.long()

            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                loss = criterion(outputs, labels.squeeze(1))
            elif criterion.__name__ == 'dice_loss':
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(true_labels.tolist())

    avg_loss = total_loss / len(val_loader.dataset)
    # 你可能需要計算 Dice 係數或其他分割指標
    # acc = accuracy_score(all_labels, all_preds)
    # print(f"Validation Accuracy: {acc}")

    return avg_loss

def main():
     # Hyperparameters and configurations
    data_dir = '/media/oldman/OldmanDoc/Document/NYCU/transformer/MRSpineSeg_Challenge_SMU/train/MR'  # Replace with your data directory
    image_size = (880, 880, 15)  # Adjust based on your data
    image_patch_size = (88, 88)  # Experiment with patch sizes
    patch_depth = 3
    num_classes = 20  # Change to the number of classes in your task
    dim = 512
    depth = 6
    heads = 8
    mlp_dim = 1024
    dropout = 0.1
    emb_dropout = 0.1
    batch_size = 2
    num_workers = 4
    learning_rate = 0.001
    epochs = 20
    shuffle = True
    val_split = 0.2  # Fraction of data for validation

    # Determine if CUDA (GPU) is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations (you might need to adjust these)
    transform = transforms.Compose([
        ##transforms.Normalize(mean=[0.5], std=[0.5])  # Simple normalization
    ])

    # Datasets and DataLoaders
    full_dataset = spinalLoader(data_dir=data_dir, target_size=image_size, transform=transform, num_classes=num_classes)
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])[0]
    val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])[1]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # No need to shuffle validation

    # Model, Optimizer, and Loss Function
    model = ViT(
        image_size=image_size,
        image_patch_size=image_patch_size,
        patch_depth=patch_depth,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Or your appropriate loss function
    # criterion = dice_loss

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = validate_model(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  Saved best model.")

    # Final Evaluation (optional)
    # If you have a separate test set, evaluate here
    # and visualize the confusion matrix
    # ... (Code for final evaluation and visualization)
if __name__ == "__main__":
    main()