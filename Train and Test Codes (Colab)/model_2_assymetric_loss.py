#BLOCK-1
#---------------------------------------------------------------------------

from google.colab import drive
import os

drive.mount('/content/drive')

os.makedirs("/content/dataset", exist_ok=True)
print("Extracting datasets...")
!unzip -q "/content/drive/MyDrive/FinalDataset_PA.zip" -d "/content/dataset"
!unzip -q "/content/drive/MyDrive/Xray-temp-master.zip" -d "/content/"
print("Extraction completed.")


#BLOCK-2
#---------------------------------------------------------------------------

# process metadata and create splits
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

csv_path = "/content/dataset/FinalDataset_PA/metadata_pa_only.csv"
df = pd.read_csv(csv_path)

# remove No Finding labels
clean_df = df[['Image Index', 'Finding Labels']].copy()
clean_df['Finding Labels'] = clean_df['Finding Labels'].apply(
    lambda x: [label for label in x.split('|') if label != "No Finding"] if isinstance(x, str) else []
)

# binarize labels
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(clean_df['Finding Labels'])
labels_df = pd.DataFrame(binary_labels, columns=mlb.classes_)
final_df = pd.concat([clean_df['Image Index'], labels_df], axis=1)

# train/val/test split (80/10/10)
train_val, test_df = train_test_split(final_df, test_size=0.10, random_state=42)
train_df, val_df = train_test_split(train_val, test_size=0.1111, random_state=42)

train_df.to_csv("/content/train_split.csv", index=False)
val_df.to_csv("/content/val_split.csv", index=False)
test_df.to_csv("/content/test_split.csv", index=False)

print(f"Total classes: {len(mlb.classes_)}")
print(f"Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


#BLOCK-3
#---------------------------------------------------------------------------

# overwrite dataset class to handle float32 labels and missing images
#(we couldn't directly change the file on Collab system so we rewrite it)
import os

PROJECT_ROOT = "/content/Xray-temp-master/Xray-temp-master"
target_file = os.path.join(PROJECT_ROOT, "data/dataset.py")

dataset_code = """
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class LargeImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (512, 512))
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx])
"""

with open(target_file, "w") as f:
    f.write(dataset_code)
print("Dataset class updated.")


#BLOCK-4
#---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
import sys
from tqdm import tqdm

PROJECT_ROOT = "/content/Xray-temp-master/Xray-temp-master"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.full_model import DenseNetCBAM
from data.dataset import LargeImageDataset

# asymmetric loss definition
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
            
        loss_pos = y * torch.log(xs_pos.clamp(min=self.eps)) * (1 - xs_pos)**self.gamma_pos
        loss_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps)) * (1 - xs_neg)**self.gamma_neg
        
        return -(loss_pos + loss_neg).mean()

# config
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_CSV = "/content/train_split.csv"
VAL_CSV = "/content/val_split.csv"
IMG_DIR = "/content/dataset/FinalDataset_PA"

# transforms
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# loaders
train_ds = LargeImageDataset(TRAIN_CSV, IMG_DIR, train_transform)
val_ds = LargeImageDataset(VAL_CSV, IMG_DIR, val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# model setup
num_classes = len(pd.read_csv(TRAIN_CSV).columns) - 1
model = DenseNetCBAM()

if hasattr(model, 'classifier'):
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
elif hasattr(model, 'fc'):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(DEVICE)

criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)
optimizer = optim.Adam(model.parameters(), lr=LR)

# train loop
os.makedirs("checkpoints", exist_ok=True)
print(f"Training on: {DEVICE} with Asymmetric Loss")
best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=True)

    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # validation
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
    # metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    tp = (all_preds * all_labels).sum(dim=0)
    fp = (all_preds * (1 - all_labels)).sum(dim=0)
    fn = ((1 - all_preds) * all_labels).sum(dim=0)
    f1_score = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Macro F1: {f1_score:.4f}")

    # save best model
    if f1_score > best_f1:
        best_f1 = f1_score
        torch.save(model.state_dict(), "checkpoints/best_model_asl.pth")
        print(">>> Best model saved.")


#BLOCK-5
#---------------------------------------------------------------------------

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

TEST_CSV = "/content/test_split.csv"
MODEL_PATH = "checkpoints/best_model_asl.pth"

# test loader
test_ds = LargeImageDataset(TEST_CSV, IMG_DIR, val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# load model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
model.to(DEVICE)

all_probs, all_labels = [], []
class_names = pd.read_csv(TEST_CSV).columns[1:].tolist()

print("Evaluating ASL model on test set...")
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        
        all_probs.append(probs)
        all_labels.append(labels.numpy())

all_probs = np.vstack(all_probs)
all_labels = np.vstack(all_labels)
all_preds = (all_probs > 0.5).astype(int)

# calculate global metrics
macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
avg_acc = accuracy_score(all_labels, all_preds)

print("\n--- OVERALL PERFORMANCE ---")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Macro AUC: {macro_auc:.4f}")
print(f"Avg Acc  : {avg_acc:.4f}\n")

# calculate per-class metrics
print("--- PER CLASS PERFORMANCE ---")
print(f"{'Pathology':<20} | {'F1':<6} | {'AUC':<6} | {'Acc':<6}")
print("-" * 47)

for i, class_name in enumerate(class_names):
    c_labels = all_labels[:, i]
    c_probs = all_probs[:, i]
    c_preds = all_preds[:, i]
    
    c_f1 = f1_score(c_labels, c_preds, zero_division=0)
    c_auc = roc_auc_score(c_labels, c_probs) if len(np.unique(c_labels)) > 1 else 0.0
    c_acc = accuracy_score(c_labels, c_preds)
    
    print(f"{class_name:<20} | {c_f1:.4f} | {c_auc:.4f} | {c_acc:.4f}")