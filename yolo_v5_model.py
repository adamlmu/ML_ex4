import os
import random
import shutil
import glob
import tarfile
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# 1) DOWNLOAD + EXTRACT OXFORD 102 FLOWERS
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
SETID_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

os.makedirs("data/flowers", exist_ok=True)
flowers_archive = os.path.join("data", "flowers", "102flowers.tgz")
setid_file = os.path.join("data", "flowers", "setid.mat")

if not os.path.exists(flowers_archive):
    print("Downloading 102 Flowers dataset...")
    r = requests.get(DATA_URL, stream=True)
    with open(flowers_archive, "wb") as f:
        f.write(r.content)
    print("Download complete.")

if not os.path.exists("data/flowers/jpg"):
    print("Extracting images...")
    with tarfile.open(flowers_archive, 'r:gz') as tar:
        tar.extractall(path="data/flowers/")
    print("Extraction complete.")

if not os.path.exists(setid_file):
    print("Downloading setid.mat...")
    r = requests.get(SETID_URL, stream=True)
    with open(setid_file, "wb") as f:
        f.write(r.content)
    print("Downloaded setid.mat.")

# 2) CREATE TWO RANDOM SPLITS: 50% TRAIN, 25% VAL, 25% TEST
def create_splits():
    all_images = sorted(glob.glob("data/flowers/jpg/*.jpg"))
    n_images = len(all_images)
    print("Total images found:", n_images)
    random.seed(42)

    for split_id in range(1, 3):
        split_dir = f"data/flowers/split{split_id}"
        # Only create the split if it doesn't already exist.
        if not os.path.exists(split_dir):
            os.makedirs(split_dir, exist_ok=True)
            # Shuffle images for each split
            random.shuffle(all_images)
            train_end = int(0.50 * n_images)
            val_end = int(0.75 * n_images)

            train_files = all_images[:train_end]
            val_files = all_images[train_end:val_end]
            test_files = all_images[val_end:]

            for phase in ["train", "val", "test"]:
                os.makedirs(os.path.join(split_dir, phase, "images"), exist_ok=True)
                os.makedirs(os.path.join(split_dir, phase, "labels"), exist_ok=True)

            for f in train_files:
                shutil.copy(f, os.path.join(split_dir, "train", "images"))
            for f in val_files:
                shutil.copy(f, os.path.join(split_dir, "val", "images"))
            for f in test_files:
                shutil.copy(f, os.path.join(split_dir, "test", "images"))

            print(f"Split {split_id}:")
            print("  Train:", len(train_files))
            print("  Val:  ", len(val_files))
            print("  Test: ", len(test_files))
        else:
            print(f"Split {split_id} already exists. Skipping split creation.")


# 3) CREATE YOLO LABELS
def simple_label_func(filename):
    idx = int(filename.split('_')[1].split('.')[0]) - 1
    label = (idx * 102) // 8189
    return label

def create_yolo_labels(img_dir, lbl_dir, class_func):
    os.makedirs(lbl_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    for img_path in images:
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        class_id = class_func(filename)
        label_file = os.path.join(lbl_dir, f"{stem}.txt")
        # Only create the label file if it doesn't exist
        if not os.path.exists(label_file):
            with open(label_file, "w") as lf:
                lf.write(f"{class_id}\n")

def create_all_labels():
    for split_id in [1, 2]:
        split_path = f"data/flowers/split{split_id}"
        for phase in ["train", "val", "test"]:
            img_dir = os.path.join(split_path, phase, "images")
            lbl_dir = os.path.join(split_path, phase, "labels")
            print(f"Creating YOLO labels for: {img_dir}")
            create_yolo_labels(img_dir, lbl_dir, simple_label_func)

# 4) DEFINE A CUSTOM DATASET FOR FLOWERS
class FlowersDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        label_path = os.path.join(self.labels_dir, f"{stem}.txt")
        with open(label_path, "r") as f:
            label_data = f.readline().strip().split()
            label = int(label_data[0])  # Extract only the class ID
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

#Define transforms for training and for validation/test
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
transform_val_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

def get_dataloaders(split_id, batch_size=32):
    split_path = f"data/flowers/split{split_id}"
    train_dataset = FlowersDataset(os.path.join(split_path, "train", "images"),
                                   os.path.join(split_path, "train", "labels"),
                                   transform=transform_train)
    val_dataset = FlowersDataset(os.path.join(split_path, "val", "images"),
                                 os.path.join(split_path, "val", "labels"),
                                 transform=transform_val_test)
    test_dataset = FlowersDataset(os.path.join(split_path, "test", "images"),
                                  os.path.join(split_path, "test", "labels"),
                                  transform=transform_val_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

# 5) DEFINE THE CLASSIFICATION MODEL
def get_model(num_classes=102):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# 6) TRAIN THE MODEL & RECORD METRICS PER EPOCH
def train_model(split_id, num_epochs=10, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    train_loader, val_loader, test_loader = get_dataloaders(split_id, batch_size)
    model = get_model(num_classes=102).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #Lists of metrics
    train_acc_history, val_acc_history, test_acc_history = [], [], []
    train_loss_history, val_loss_history, test_loss_history = [], [], []

    for epoch in range(num_epochs):
        #Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        #Validation Phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        epoch_val_loss = running_loss / total
        epoch_val_acc = correct / total
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        #Test Phase
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        epoch_test_loss = running_loss / total
        epoch_test_acc = correct / total
        test_loss_history.append(epoch_test_loss)
        test_acc_history.append(epoch_test_acc)

        print(f"Split {split_id}, Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}")
        print(f"  Val   Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
        print(f"  Test  Loss: {epoch_test_loss:.4f}, Acc: {epoch_test_acc:.4f}")

    #Plots
    epochs_range = range(1, num_epochs+1)
    plt.figure(figsize=(12, 5))

    #Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_history, label="Train Accuracy",marker="o")
    plt.plot(epochs_range, val_acc_history, label="Validation Accuracy",marker="o")
    plt.plot(epochs_range, test_acc_history, label="Test Accuracy",marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training, Validation & Test Accuracy Over Epochs (Split {split_id})")
    plt.legend()
    plt.grid()

    #Cross-Entropy Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_history, label="Training Loss",marker="o")
    plt.plot(epochs_range, val_loss_history, label="Validation Loss",marker="o")
    plt.plot(epochs_range, test_loss_history, label="Test Loss",marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Training, Validation & Test loss Over Epochs (Split {split_id})")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# 7) MAIN
if __name__ == "__main__":
    #Create splits
    create_splits()
    create_all_labels()

    #Train and plot for all splits
    for split in [1, 2]:
        train_model(split_id=split, num_epochs=10, batch_size=32)
