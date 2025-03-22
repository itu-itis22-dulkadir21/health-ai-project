import cv2
import numpy as np
import matplotlib.pyplot as plt

def denoise_ct_image(image_path):
    # Görüntüyü yükleme
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Non-Local Means Denoising
    denoised_nlm = cv2.fastNlMeansDenoising(
        src=img,
        h=10,           # Gürültü şiddeti (CT için tipik: 10-20)
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    # 2. Median Blur + CLAHE
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    denoised_clahe = clahe.apply(cv2.medianBlur(img, 3))
    
    # 3. Wiener Filtresi (Scikit-image ile)
    from skimage.restoration import wiener
    psf = np.ones((3, 3)) / 9  # Point spread function
    denoised_wiener = wiener(img, psf, 1.5)
    
    return denoised_nlm, denoised_clahe, denoised_wiener

# Kullanım
nlm, clahe, wiener = denoise_ct_image("C:/Kanama Veri Seti/PNG/10002.png")

# Görselleştirme
plt.figure(figsize=(20,10))
plt.subplot(141), plt.imshow(cv2.imread("C:/Kanama Veri Seti/PNG/10002.png", cv2.IMREAD_GRAYSCALE), cmap='gray'), plt.title('Original')
plt.subplot(142), plt.imshow(nlm, cmap='gray'), plt.title('NLM Denoised')
plt.subplot(143), plt.imshow(clahe, cmap='gray'), plt.title('CLAHE+Median')
plt.subplot(144), plt.imshow(wiener, cmap='gray'), plt.title('Wiener Filter')
plt.show()