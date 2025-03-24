import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm  # pip install timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Hiperparametreler
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
IMAGE_SIZE = 224  # Swin Transformer için yaygın input boyutu

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Veri ön işleme (transformations)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# CSV üzerinden veri okuyan özel dataset
class CSVDataset(Dataset):
    def _init_(self, csv_file, root_dir, transform=None):
        """
        csv_file: 'name number', 'section' ve 'label' sütunlarını içeren CSV dosyasının yolu.
        root_dir: Görüntülerin bulunduğu ana dizin (burada "total" klasörü).
        transform: Uygulanacak ön işleme transformasyonları.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # CSV sütunlarını kontrol et (debug için)
        print("CSV Columns:", self.data_frame.columns)

    def _len_(self):
        return len(self.data_frame)
    
    def _getitem_(self, idx):
        row = self.data_frame.iloc[idx]
        # "name number" sütunu, dosya adını içeriyor (örneğin, "10000.png")
        img_path = os.path.join(self.root_dir, row['img_name'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row['label'])
        # Bölüm bilgisi isteğe bağlı olarak kullanılabilir
        section = row['section'] if 'section' in row.index else None
        return image, label, section

# Dataset yolları: PNG dosyaları artık "total" klasöründe
data_dir = "/kaggle/input/brain-stroke-ds/dataset/all"
val_csv_path = "/kaggle/input/real-data/test_section_based.csv"  # CSV dosyanızın tam yolu
train_csv_path = "/kaggle/input/real-data/train_section_based.csv"  # CSV dosyanızın tam yolu

# Eğitim ve doğrulama dataset'lerini oluşturuyoruz.
train_dataset = CSVDataset(csv_file=train_csv_path, root_dir=data_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dataset = CSVDataset(csv_file=val_csv_path, root_dir=data_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model: timm kütüphanesinden Swin Transformer kullanımı
num_classes = 2  # inme var: 1, inme yok: 0
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
model.to(device)

# Loss fonksiyonu
criterion = nn.CrossEntropyLoss()

# Optimizer: AdamW (momentum etkisini betas ile taşır) ve weight decay
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

# Learning rate scheduler: CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Eğitim ve doğrulama döngüsü
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _meta in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / total
    train_acc = correct / total * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels, _meta in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_loss = val_loss / total_val
    val_acc = correct_val / total_val * 100
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}: LR: {current_lr:.6f} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    scheduler.step()

print("Training completed.")
# Modeli kaydet
model_path = "/kaggle/working/swin_stroke_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Eğitim sonuçlarını grafikle gösterelim:
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, NUM_EPOCHS+1), train_losses, label="Train Loss")
plt.plot(range(1, NUM_EPOCHS+1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Eğrisi")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, NUM_EPOCHS+1), train_accuracies, label="Train Acc")
plt.plot(range(1, NUM_EPOCHS+1), val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Eğrisi")
plt.legend()
plt.show()

# EK PERFORMANS METRİKLERİ: Confusion Matrix ve Classification Report
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels, _meta in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
