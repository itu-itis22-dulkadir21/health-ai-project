import cv2
import numpy as np
import pydicom
import os
from skimage import exposure
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. DICOM veya Standart Görüntü Yükleme
def load_image(path):
    if path.endswith('.dcm'):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 0-255 aralığına ölçeklendir
    else:  # PNG/JPG
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

# 2. Gürültü Azaltma ve Kontrast İyileştirme
def preprocess_ct_image(img):
    # Gürültü azaltma (Non-Local Means)
    denoised = cv2.fastNlMeansDenoising(
        img,
        h=15,                  # CT için optimize edilmiş parametre
        templateWindowSize=7,  # Küçük dokular için 7x7
        searchWindowSize=21    # Geniş arama alanı
    )
    
    # Kontrast iyileştirme (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Gamma düzeltme (Opsiyonel)
    gamma = 1.2
    gamma_corrected = np.power(enhanced / 255.0, gamma) * 255.0
    return gamma_corrected.astype(np.uint8)

# 3. Veri Artırma (Data Augmentation)
def augment_data(images, batch_size=8):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0  # CT'de siyah arka plan için
    )
    
    aug_iter = datagen.flow(
        np.expand_dims(images, axis=-1),  # Keras için channel dimension ekle
        batch_size=batch_size,
        shuffle=True
    )
    return aug_iter

# 4. Patch Oluşturma (Model Input Boyutu İçin)
def generate_patches(img, patch_size=128, stride=64):
    patches = []
    height, width = img.shape
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

# 5. Ana İşlem Akışı
if __name__ == "__main__":
    # Parametreler
    input_path = "10187.png"  # veya "ct_scan.png"
    output_dir = "preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    # Görüntüyü yükle ve ön işle
    raw_img = load_image(input_path)
    processed_img = preprocess_ct_image(raw_img)

    # Görselleştirme
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(raw_img, cmap='gray'), plt.title('Ham Görüntü')
    plt.subplot(122), plt.imshow(processed_img, cmap='gray'), plt.title('İşlenmiş Görüntü')
    plt.show()

    # Patch'leri oluştur ve kaydet
    patches = generate_patches(processed_img)
    for i, patch in enumerate(patches):
        cv2.imwrite(os.path.join(output_dir, f"patch_{i}.png"), patch)

    # Veri Artırma Örneği
    sample_images = np.expand_dims(processed_img, axis=0)  # Batch boyutu ekle
    aug_generator = augment_data(sample_images)
    
    # Artırılmış görüntüleri göster
    batch = next(aug_generator)
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(batch[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle('Artırılmış Örnekler')
    plt.show()