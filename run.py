import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from model_architecture import SwinStrokeModel

def main():
    # GPU Kontrolü
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model Yükleme
    model = SwinStrokeModel(
        model_name='swin_base_patch4_window7_224',
        pretrained=False,
        num_classes=2
    ).to(device)
    
    # StateDict'i düzelt
    state_dict = torch.load("swin_stroke_model.pth", map_location=device)
    
    # Anahtar isimlerini düzelt: "swin." önekini ekle
    fixed_state_dict = {}
    for key, value in state_dict.items():
        fixed_key = f"swin.{key}"  # Model içindeki swin alt modülüne göre
        fixed_state_dict[fixed_key] = value
    
    model.load_state_dict(fixed_state_dict, strict=True)
    model.eval()

    # Dataset ve DataLoader (Aynı kalıyor)
    class StrokeTestDataset(Dataset):
        def __init__(self, csv_path, transform=None):
            self.df = pd.read_csv(csv_path)
            self.transform = transform
            
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            img_name = self.df.iloc[idx]['img_name']
            section = self.df.iloc[idx]['section']
            img_path = f"test-data/{img_name}"
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_name, section

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = StrokeTestDataset("test-data-name-class.csv", transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # Tahmin İşlemi
    results = []
    with torch.no_grad():
        for images, img_names, sections in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            batch_results = [
                [img_names[i], sections[i], int(preds[i])]
                for i in range(len(img_names))
            ]
            results.extend(batch_results)
    
    # CSV'ye Kaydet
    pd.DataFrame(results, columns=["img_name", "section", "label"]).to_csv("predictions.csv", index=False)
    print("Tahminler kaydedildi!")

if __name__ == '__main__':
    main()