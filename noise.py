import cv2
import numpy as np
from matplotlib import pyplot as plt

def add_multiple_noise_to_ct(image, 
                            gaussian_sigma=10,      # 5-15 aralığında (orta değer)
                            salt_pepper_ratio=0.03, # %1-5 aralığında (orta değer)
                            speckle_var=0.01,       # Düşük speckle varyansı
                            poisson_scale=0.5):     # Ölçeklendirilmiş Poisson
    """
    Beyin BT görüntülerine kontrollü gürültü türleri ekler.
    Gauss için 5-15, Salt & Pepper için %1-5 aralığına göre ayarlanmıştır.
    
    Args:
        image: Giriş görüntüsü (grayscale veya renkli)
        gaussian_sigma: Gauss gürültüsü standart sapması (5-15 arası)
        salt_pepper_ratio: Salt & Pepper gürültüsü yoğunluğu (0.01-0.05 arası)
        speckle_var: Speckle gürültüsü varyansı 
        poisson_scale: Poisson gürültüsü şiddeti
        
    Returns:
        all_results: Orijinal ve gürültülü görüntüleri içeren sözlük
    """
    # Görüntü tipini ve boyutlarını al
    if len(image.shape) == 2:
        # Gri tonlamalı görüntü
        row, col = image.shape
        ch = 1
        is_gray = True
    else:
        # Renkli görüntü
        row, col, ch = image.shape
        is_gray = False
    
    # Görüntüyü float formatına çevir (hesaplamalar için)
    img_float = image.astype(np.float32)
    
    # Sonuçları saklamak için sözlük
    all_results = {
        'original': image.copy()
    }
    
    # A) Gauss Gürültüsü (5-15 aralığında sigma)
    gauss_noise = np.random.normal(0, gaussian_sigma, (row, col, ch) if not is_gray else (row, col))
    if is_gray:
        gauss_noisy = img_float + gauss_noise
    else:
        gauss_noise = gauss_noise.reshape(row, col, ch)
        gauss_noisy = img_float + gauss_noise
    
    # Değerleri 0-255 aralığında tut
    gauss_noisy = np.clip(gauss_noisy, 0, 255).astype(np.uint8)
    all_results['gaussian'] = gauss_noisy
    
    # B) Salt & Pepper Gürültüsü (%1-5 aralığında)
    s_p_noise = image.copy()
    # Salt (beyaz noktalar)
    salt_coords = [np.random.randint(0, i - 1, int(salt_pepper_ratio * image.size * 0.5)) for i in image.shape[:2]]
    if is_gray:
        s_p_noise[salt_coords[0], salt_coords[1]] = 255
    else:
        s_p_noise[salt_coords[0], salt_coords[1], :] = 255
    
    # Pepper (siyah noktalar)
    pepper_coords = [np.random.randint(0, i - 1, int(salt_pepper_ratio * image.size * 0.5)) for i in image.shape[:2]]
    if is_gray:
        s_p_noise[pepper_coords[0], pepper_coords[1]] = 0
    else:
        s_p_noise[pepper_coords[0], pepper_coords[1], :] = 0
        
    all_results['salt_pepper'] = s_p_noise
    
    # C) Speckle Gürültüsü (I + I*N formülü)
    if is_gray:
        speckle_noise = np.random.randn(row, col) * speckle_var
        speckle_noisy = img_float + img_float * speckle_noise
    else:
        speckle_noise = np.random.randn(row, col, ch) * speckle_var
        speckle_noisy = img_float + img_float * speckle_noise
    
    speckle_noisy = np.clip(speckle_noisy, 0, 255).astype(np.uint8)
    all_results['speckle'] = speckle_noisy
    
    # D) Poisson Gürültüsü
    # Görüntüyü normalize et (0-1 aralığı)
    img_norm = img_float / 255.0
    
    # Poisson gürültüsü ekle (ölçeklendirme kullanarak)
    poisson_noisy = np.random.poisson(img_norm * poisson_scale * 255) / (poisson_scale * 255)
    poisson_noisy = np.clip(poisson_noisy * 255, 0, 255).astype(np.uint8)
    all_results['poisson'] = poisson_noisy
    
    # Tüm gürültü türlerinin birleşimi
    combined = img_float.copy()
    
    # Gauss 
    gauss_component = np.random.normal(0, gaussian_sigma * 0.7, (row, col, ch) if not is_gray else (row, col))
    if not is_gray:
        gauss_component = gauss_component.reshape(row, col, ch)
    combined += gauss_component
    
    # Salt & Pepper
    sp_coords = [np.random.randint(0, i - 1, int(salt_pepper_ratio * 0.7 * image.size)) for i in image.shape[:2]]
    if is_gray:
        sp_mask = np.zeros((row, col))
        sp_mask[sp_coords[0], sp_coords[1]] = np.random.choice([0, 255], len(sp_coords[0]))
        combined += (sp_mask - img_float) * 0.5
    else:
        for i in range(ch):
            sp_mask = np.zeros((row, col))
            sp_mask[sp_coords[0], sp_coords[1]] = np.random.choice([0, 255], len(sp_coords[0]))
            combined[:,:,i] += (sp_mask - img_float[:,:,i]) * 0.5
    
    # Speckle
    if is_gray:
        speckle_component = img_float * np.random.randn(row, col) * speckle_var * 0.7
    else:
        speckle_component = img_float * np.random.randn(row, col, ch) * speckle_var * 0.7
    combined += speckle_component
    
    # Poisson
    poisson_component = np.random.poisson(img_norm * poisson_scale * 0.7 * 255) / (poisson_scale * 0.7 * 255) * 255 - img_float
    combined += poisson_component * 0.5
    
    # Son görüntüyü düzelt
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    all_results['combined'] = combined
    
    return all_results

def visualize_noise_effects(results):
    """
    Gürültü ekleme sonuçlarını görselleştirir
    """
    # Görüntüleri göster
    plt.figure(figsize=(15, 12))
    
    titles = ["Orijinal Görüntü", "Gauss Gürültüsü (5-15)", "Salt & Pepper (%1-5)", 
              "Speckle Gürültüsü", "Poisson Gürültüsü", "Birleşik Gürültüler"]
    images = [results['original'], results['gaussian'], results['salt_pepper'], 
              results['speckle'], results['poisson'], results['combined']]
    
    for i in range(len(images)):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('bt_gurultu_karsilastirma.png', dpi=300)
    plt.show()

# Kullanım örneği - farklı gürültü seviyeleri gösterimi
def test_noise_levels(image):
    """
    Farklı gürültü seviyelerini test eder ve görselleştirir
    """
    # Gauss için farklı sigma değerleri (5-15 aralığında)
    gauss_sigmas = [5, 10, 15]
    
    # Salt & Pepper için farklı oranlar (%1-5 aralığında)
    sp_ratios = [0.01, 0.03, 0.05]
    
    # Sonuçları göster
    plt.figure(figsize=(15, 10))
    
    # Orijinal görüntü
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title("Orijinal Görüntü")
    plt.axis('off')
    
    # Farklı Gauss sigma değerleri
    for i, sigma in enumerate(gauss_sigmas, 1):
        result = add_multiple_noise_to_ct(image, gaussian_sigma=sigma, salt_pepper_ratio=0.03)
        plt.subplot(3, 3, i+1)
        plt.imshow(result['gaussian'], cmap='gray' if len(image.shape) == 2 else None)
        plt.title(f"Gauss (σ={sigma})")
        plt.axis('off')
    
    # Farklı Salt & Pepper oranları
    for i, ratio in enumerate(sp_ratios, 1):
        result = add_multiple_noise_to_ct(image, gaussian_sigma=10, salt_pepper_ratio=ratio)
        plt.subplot(3, 3, i+4)
        plt.imshow(result['salt_pepper'], cmap='gray' if len(image.shape) == 2 else None)
        plt.title(f"Salt & Pepper ({ratio*100:.0f}%)")
        plt.axis('off')
    
    # Kombine gürültü örnekleri
    for i, (sigma, ratio) in enumerate(zip([5, 10, 15], [0.01, 0.03, 0.05]), 1):
        result = add_multiple_noise_to_ct(image, gaussian_sigma=sigma, salt_pepper_ratio=ratio)
        plt.subplot(3, 3, i+7)
        plt.imshow(result['combined'], cmap='gray' if len(image.shape) == 2 else None)
        plt.title(f"Kombine (σ={sigma}, SP={ratio*100:.0f}%)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('bt_gurultu_seviyeleri.png', dpi=300)
    plt.show()
    
def save_noisy_image(image, filename="combined_noise_image.png", path="./"):
    """
    Gürültülü görüntüyü belirtilen dosya adı ve yola kaydeder.
    
    Args:
        image: Kaydedilecek görüntü
        filename: Dosya adı (varsayılan: combined_noise_image.png)
        path: Kayıt yolu (varsayılan: geçerli dizin)
    
    Returns:
        save_path: Tam kayıt yolu
    """
    import os
    
    # Tam dosya yolunu oluştur
    save_path = os.path.join(path, filename)
    
    # Görüntüyü kaydet
    cv2.imwrite(save_path, image)
    print(f"Görüntü başarıyla kaydedildi: {save_path}")
    
    return save_path

# Ana kullanım örneği
if __name__ == "__main__":
    # Beyin BT görüntüsünü yükle
    img = cv2.imread('deney.png', 0)  # Gri tonlamalı olarak oku (BT için uygun)
    
    if img is None:
        print("Görüntü yüklenemedi! Dosya yolunu kontrol edin.")
    else:
        # 1. Standart gürültü ekleme ve görselleştirme
        results = add_multiple_noise_to_ct(
            img,
            gaussian_sigma=10,      # 5-15 aralığında orta değer
            salt_pepper_ratio=0.03, # %1-5 aralığında orta değer
            speckle_var=0.01,       
            poisson_scale=0.5       
        )
        
        # 2. Tüm gürültü türlerini tek pencerede görselleştirme ve kaydetme
        visualize_noise_effects(results)  # Bu fonksiyon zaten 'bt_gurultu_karsilastirma.png' kaydediyor
        
        # 3. Sadece birleşik gürültüyü ayrı bir pencerede gösterme ve kaydetme
        cv2.imshow("Birleşik Gürültü", results['combined'])  # Birleşik gürültüyü göster
        cv2.waitKey(0)  # Görüntülerin kapanmasını bekler
        
        # 4. Birleşik gürültüyü ayrı bir PNG dosyası olarak kaydetme
        save_noisy_image(results['combined'], filename="combined_noise_output.png", path="./")
        
        # 5. Farklı gürültü seviyelerini test et ve görselleştir
        test_noise_levels(img)