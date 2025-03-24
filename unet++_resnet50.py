# -*- coding: utf-8 -*-
"""Kaggle_UNetPlusPlus_ResNet50_Hybrid_Model_CSV.ipynb"""

# Gerekli Kütüphaneler
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import torchvision.models as models

# GPU Kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

########################################
# 1. CSV Destekli Veri Seti Sınıfı (Aynı)
########################################

class StrokeCSVDataset(Dataset):
    def __init__(self, csv_path, img_size=256):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        
        self.section_encoder = LabelEncoder()
        self.df['section_encoded'] = self.section_encoder.fit_transform(self.df['section'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = "/kaggle/input/brain-stroke-ds/dataset/all/" + self.df.iloc[idx]['img_name']
        label = self.df.iloc[idx]['label']
        section = self.df.iloc[idx]['section_encoded']
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"{img_path} okunamadı!")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255.0
        
        return (
            torch.from_numpy(image).permute(2, 0, 1).float(),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(section, dtype=torch.long)
        )

########################################
# 2. ResNet50 Tabanlı UNet++ Modeli
########################################

class UNetPlusPlusResNet(nn.Module):
    def __init__(self, num_sections):
        super().__init__()
        
        # ResNet50 Encoder
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Bottleneck (ResNet50 için 2048 kanal giriş)
        self.bottleneck = self._block(2048, 2048)
        
        # Decoder (Kanallar ResNet50'e göre ayarlandı)
        self.up1 = self._upblock(2048, 1024)
        self.up2 = self._upblock(1024, 512)
        self.up3 = self._upblock(512, 256)
        self.up4 = self._upblock(256, 128)
        self.up5 = self._upblock(128, 64)
        
        # Çıkış katmanı
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        
        # Sınıflandırıcı (2048 kanal giriş)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Kesit sınıflandırıcı (2048 kanal giriş)
        self.section_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_sections),
            nn.Softmax(dim=1)
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def _upblock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            self._block(out_ch, out_ch)
        )
        
    def forward(self, x):
        # Encoder
        x = self.encoder[0](x)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        x = self.encoder[4](x)
        x = self.encoder[5](x)
        x = self.encoder[6](x)
        e4 = self.encoder[7](x)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder
        d1 = self.up1(b)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
        d4 = self.up4(d3)
        d5 = self.up5(d4)
        
        # Çıkış
        reconstructed = torch.sigmoid(self.out(d5))
        
        # Sınıflandırma
        cls_output = self.classifier(b)
        section_output = self.section_classifier(b)
        
        return reconstructed, cls_output.squeeze(), section_output

########################################
# 3. Geliştirilmiş Eğitim Döngüsü (Aynı)
########################################

# Dataset ve DataLoader
train_csv = "/kaggle/input/brain-stroke-ds/dataset/train_stratified.csv"
test_csv = "/kaggle/input/brain-stroke-ds/dataset/test_stratified.csv"

train_dataset = StrokeCSVDataset(train_csv)
test_dataset = StrokeCSVDataset(test_csv)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# Model ve Optimizasyon
num_sections = len(train_dataset.section_encoder.classes_)
model = UNetPlusPlusResNet(num_sections).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# Loss Fonksiyonları
recon_criterion = nn.MSELoss()
cls_criterion = nn.BCELoss()
section_criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, sections in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        sections = sections.to(device)
        
        optimizer.zero_grad()
        recon_output, cls_output, section_output = model(images)
        
        # Loss hesapla
        loss_recon = recon_criterion(recon_output, images)
        loss_cls = cls_criterion(cls_output, labels)
        loss_section = section_criterion(section_output, sections)
        loss = 0.5 * loss_recon + 0.3 * loss_cls + 0.2 * loss_section  # Ağırlıklı kombinasyon
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (cls_output > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels, sections in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            sections = sections.to(device)
            
            recon_output, cls_output, section_output = model(images)
            loss_recon = recon_criterion(recon_output, images)
            loss_cls = cls_criterion(cls_output, labels)
            loss_section = section_criterion(section_output, sections)
            loss = 0.5 * loss_recon + 0.3 * loss_cls + 0.2 * loss_section
            
            val_loss += loss.item()
            predicted = (cls_output > 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    # Metrikleri hesapla
    train_acc = 100 * correct / total
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss/len(test_loader):.4f} | Val Acc: {val_acc:.2f}%")
    print("-"*50)
    
    # En iyi modeli kaydet
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_unetplusplus_model.pth")

########################################
# 4. Test ve Değerlendirme
########################################

# Modeli yükle
model.load_state_dict(torch.load("best_unetplusplus_model.pth"))
model.eval()

# Test işlemleri
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.cpu().numpy()
        
        _, cls_output, _ = model(images)
        preds = (cls_output.cpu().numpy() > 0.5).astype(int)
        
        all_preds.extend(preds)
        all_labels.extend(labels)

# Performans raporu
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# ROC Eğrisi
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()