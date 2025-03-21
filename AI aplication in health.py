import numpy as np
import os
import cv2
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

"""TASK-A

1. HOG fonksiyonu (görüntüden özellik çıkarma)
2. Görüntü yükleme fonksiyonu (load_ct_images)
3. K-means fonksiyonu (kümeleme)
4. Görselleştirme fonksiyonları

"""

def hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), normalise=False):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.
    Compute a Histogram of Oriented Gradients (HOG) by
        1. (optional) global image normalisation
        2. computing the gradient image in x and y
        3. computing gradient histograms
        4. normalising across blocks
        5. flattening into a feature vector
    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.
    Returns
    -------
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.
    """
    image = np.atleast_2d(image)
    """
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if image.ndim > 3:
        raise ValueError("Currently only supports grey-level images")

    if normalise:
        image = sqrt(image)
    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)
    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    magnitude = sqrt(gx ** 2 + gy ** 2)
    orientation = arctan2(gy, (gx + 1e-15)) * (180 / pi) + 90
    sx, sy = image.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, 0)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, 0)
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, magnitude, 0)
        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx/2)::cx, int(cy/2)::cy]
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksx, n_blocksy,
                                  bx, by, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[x:x + bx, y:y + by, :]
            eps = 1e-5
            normalised_blocks[x, y, :] = block / sqrt(block.sum() ** 2 + eps)

    return normalised_blocks.ravel()


"""BURAYA KADAR OLAN KISIM HOG FONKSIYONU"""    
    

"""GÖRÜNTÜ YÜKLEME FONKSIYONU"""
def load_ct_images(ct_directory, resize=True, target_size=(256, 256), enhance_contrast=True):
    """
    Load all CT images from a directory
    
    Parameters:
    -----------
    ct_directory : str
        Directory containing CT images
    resize : bool
        Whether to resize images to standard size
    target_size : tuple
        Size to resize images to if resize=True
    enhance_contrast : bool
        Whether to apply histogram equalization
        
    Returns:
    --------
    images : list
        List of loaded CT images as numpy arrays
    image_paths : list
        List of paths to the loaded images
    """
    # Check if directory exists
    if not os.path.exists(ct_directory):
        print(f"Error: Directory {ct_directory} does not exist")
        return [], []
    
    # Get all image files
    image_paths = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.dcm']
    
    for filename in os.listdir(ct_directory):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in valid_extensions:
            image_paths.append(os.path.join(ct_directory, filename))
    
    if not image_paths:
        print(f"No image files found in {ct_directory}")
        return [], []
    
    print(f"Found {len(image_paths)} CT images in {ct_directory}")
    
    # Load images
    images = []
    for img_path in image_paths:
        # Load image (grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Error: Could not load image {img_path}")
            continue
        
        # Resize if requested
        if resize:
            img = cv2.resize(img, target_size)
        
        # Enhance contrast if requested
        if enhance_contrast:
            img = cv2.equalizeHist(img)
        
        images.append(img)
    
    print(f"Successfully loaded {len(images)} CT images")
    return images, image_paths

""" KMEANS FONKSIYONU"""

def kmeans(dataset, max_iterations=100, epsilon=0.001, distance='euclidean'):
    """
    K-means clustering algorithm optimized for brain CT image analysis using 3 clusters.
    Typically represents: normal tissue, potentially abnormal tissue, and background.
    
    Parameters:
    -----------
    dataset : ndarray
        Input dataset of shape (n_samples, n_features)
    max_iterations : int
        Maximum number of iterations to prevent infinite loops
    epsilon : float
        Convergence threshold
    distance : str
        Distance metric to use (currently only supports 'euclidean')
        
    Returns:
    --------
    centroids : ndarray
        Final cluster centroids
    history_centroids : list
        List of centroids at each iteration
    labels : ndarray
        Cluster assignments for each data point
    """
    # Fixed number of clusters for brain CT analysis
    k = 3
    
    # Initialize
    num_instances, num_features = dataset.shape
    centroids = dataset[np.random.choice(num_instances, k, replace=False)]
    history_centroids = [centroids.copy()]
    
    # Use 1D array for cluster assignments
    labels = np.zeros(num_instances, dtype=int)
    
    # Main loop
    for iteration in range(max_iterations):
        old_centroids = centroids.copy()
        
        # Assign points to nearest centroid
        for i, point in enumerate(dataset):
            # Calculate distance to each centroid
            distances = np.array([np.linalg.norm(point - centroid) for centroid in centroids])
            labels[i] = np.argmin(distances)
        
        # Update centroids
        new_centroids = np.zeros((k, num_features))
        for cluster in range(k):
            cluster_points = dataset[labels == cluster]
            if len(cluster_points) > 0:
                new_centroids[cluster] = cluster_points.mean(axis=0)
            else:
                # Handle empty clusters - reinitialize from farthest point
                distances = np.array([np.linalg.norm(dataset - centroids[j], axis=1) 
                                     for j in range(k)])
                distances = distances.sum(axis=0)
                new_centroids[cluster] = dataset[np.argmax(distances)]
        
        centroids = new_centroids
        history_centroids.append(centroids.copy())
        
        # Check convergence
        if np.linalg.norm(centroids - old_centroids) < epsilon:
            break
    
    return centroids, history_centroids, labels


"""VİSUALİZİNG"""
def visualize_anatomical_clusters(features, labels, centroids, feature_reduction=True):
    """
    Visualize brain CT clustering results by anatomical regions.
    
    Parameters:
    -----------
    features : ndarray
        HOG features or other high-dimensional data
    labels : ndarray
        Cluster assignments from K-means
    centroids : ndarray
        Final centroids from K-means
    feature_reduction : bool
        Whether to use PCA for dimensionality reduction
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # Anatomical region colors and labels
    colors = ['#3498DB', '#AF7AC5', '#28B463']  # Blue, Purple, Green
    cluster_names = ['Eye Regions', 'Nasal/Sinus Areas', 'Brain Tissue']
    
    # Reduce dimensions for visualization if needed
    if feature_reduction and features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        centroids_2d = pca.transform(centroids)
    else:
        features_2d = features
        centroids_2d = centroids
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    for i in range(3):  # Always 3 clusters
        cluster_points = features_2d[labels == i]
        plt.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            s=50, 
            c=colors[i],
            label=cluster_names[i]
        )
    
    # Plot centroids
    plt.scatter(
        centroids_2d[:, 0], 
        centroids_2d[:, 1], 
        s=200, 
        c='black', 
        marker='X', 
        label='Centroids'
    )
    
    plt.title('CT Image Anatomical Segmentation')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def visualize_segmented_ct(image, labels, centroids):
    """
    Visualize anatomical segmentation directly on CT image.
    
    Parameters:
    -----------
    image : ndarray
        Original CT image
    labels : ndarray
        Cluster assignments from K-means
    centroids : ndarray
        Final cluster centroids
    """
    # Create a color-mapped segmentation mask
    h, w = image.shape
    segmentation = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color map for anatomical regions
    colors = [
        [52, 152, 219],   # Blue - Eye regions
        [175, 122, 197],  # Purple - Nasal/Sinus areas
        [40, 180, 99]     # Green - Brain tissue
    ]
    
    # Map HOG feature clusters back to image segments
    # This requires careful implementation based on how HOG features were extracted
    # A simplified approach would be to compute HOG for each pixel region
    # and assign it to the closest centroid
    
    # Display original and segmented images
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original CT Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation)
    plt.title('Anatomical Segmentation')
    plt.axis('off')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(colors[0])/255, label='Eye Regions'),
        Patch(facecolor=np.array(colors[1])/255, label='Nasal/Sinus Areas'),
        Patch(facecolor=np.array(colors[2])/255, label='Brain Tissue')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()


"""TASK-B"""

import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ResNet50 modelini yükle - son sınıflandırma katmanını çıkar
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
# Son katmanı kaldırıp, önceki katmanın çıktısını almak için
model = torch.nn.Sequential(*list(model.children())[:-1])  
model.eval()

# Görüntü ön işleme fonksiyonu
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')  # Görüntüyü RGB formatında aç
    img_tensor = transform(img).unsqueeze(0)  # Batch boyutu ekle
    return img_tensor

# Öznitelik çıkarma fonksiyonu
def extract_features(img_path):
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():  # Gradyan hesaplama kapalı
        features = model(img_tensor)
    return features.squeeze().numpy().flatten()  # 2048 boyutlu vektör haline getir
  # Tensörü numpy dizisine çevir

# Kendi görüntü yükleme fonksiyonunuzu kullanın
ct_directory = input("BT görüntülerinin bulunduğu dizini girin: ")
_, image_paths = load_ct_images(ct_directory)

if not image_paths:
    print("Görüntü bulunamadı veya yüklenemedi.")
    exit()

# Tüm görsellerden öznitelikleri çıkar
features = np.array([extract_features(path) for path in image_paths])

# DBSCAN ile kümeleme
dbscan = DBSCAN(eps=5, min_samples=2)
clusters = dbscan.fit_predict(features)

# Sonuçları görselleştir
plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis')
plt.title("ResNet50 + DBSCAN Kümeleme Sonuçları")
plt.show()

# Kümeleri yazdır
for i, cluster in enumerate(clusters):
    print(f"Görsel {image_paths[i]} -> Küme {cluster}")



"""TASK-C"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.cluster.hierarchy import fcluster, linkage
import seaborn as sns

# 📌 1️⃣ Görüntüleri Yükleme ve Gri Tonlamaya Çevirme
def load_gray_images(ct_directory, target_size=(256, 256)):
    """BT görüntülerini yükleyip gri tonlamaya çevirir."""
    image_paths = []
    images = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.dcm']

    for filename in os.listdir(ct_directory):
        if os.path.splitext(filename)[1].lower() in valid_extensions:
            img_path = os.path.join(ct_directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlama
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img)
                image_paths.append(img_path)

    print(f"Toplam {len(images)} görüntü yüklendi.")
    return images, image_paths

# 📌 2️⃣ SSIM Skor Matrisi Hesaplama
def compute_ssim_matrix(images):
    """Görseller arasındaki SSIM skor matrisini oluşturur."""
    num_images = len(images)
    ssim_matrix = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(i, num_images):
            if i == j:
                ssim_matrix[i, j] = 1.0  # Kendisiyle benzerlik 1.0 olmalı
            else:
                score, _ = ssim(images[i], images[j], full=True)
                ssim_matrix[i, j] = score
                ssim_matrix[j, i] = score  # Simetrik matris

    return ssim_matrix

# 📌 3️⃣ Dynamic Threshold Selection
def find_optimal_threshold(ssim_matrix, alpha=0.5):
    """Dinamik olarak SSIM tabanlı en iyi threshold'u belirler."""
    mean_ssim = np.mean(ssim_matrix)
    std_ssim = np.std(ssim_matrix)
    optimal_threshold = mean_ssim + alpha * std_ssim
    return optimal_threshold

# 📌 4️⃣ SSIM Benzerliklerine Göre Kümeleme
def cluster_images(ssim_matrix, threshold):
    """SSIM skorlarına göre görüntüleri gruplar."""
    linked = linkage(1 - ssim_matrix, method='ward')  # 1 - SSIM = mesafe
    clusters = fcluster(linked, threshold, criterion='distance')  # Eşik bazlı kümeleme
    return clusters

# 📌 5️⃣ SSIM Isı Haritasını Görselleştirme ve Küme Gösterimi
def visualize_clusters(images, image_paths, clusters, ssim_matrix):
    """Kümeleme sonuçlarını görselleştirir."""
    # Isı haritası göster
    plt.figure(figsize=(10, 8))
    sns.heatmap(ssim_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=[os.path.basename(p) for p in image_paths],
                yticklabels=[os.path.basename(p) for p in image_paths])
    plt.title("SSIM Benzerlik Matrisi")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Kümeleri görselleştir
    cluster_dict = {}
    for i, (path, cluster) in enumerate(zip(image_paths, clusters)):
        cluster_dict.setdefault(cluster, []).append((i, os.path.basename(path)))
    
    # Her küme için örnek görüntüleri göster
    for cluster_id, items in cluster_dict.items():
        n_images = min(5, len(items))  # En fazla 5 görüntü göster
        if n_images > 0:
            plt.figure(figsize=(12, 3))
            plt.suptitle(f"Küme {cluster_id}")
            
            for i in range(n_images):
                idx, name = items[i]
                plt.subplot(1, n_images, i+1)
                plt.imshow(images[idx], cmap='gray')
                plt.title(name, fontsize=8)
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()

# Ana işlev
ct_directory = input("BT görüntülerinin bulunduğu dizini girin: ")
images, image_paths = load_gray_images(ct_directory)

if images:
    # SSIM matrisini hesapla
    ssim_matrix = compute_ssim_matrix(images)
    
    # Dinamik Threshold Belirleme
    optimal_threshold = find_optimal_threshold(ssim_matrix, alpha=0.5)
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    # Threshold ile Kümeleme Yap
    clusters = cluster_images(ssim_matrix, threshold=optimal_threshold)
    
    # Sonuçları görselleştir
    visualize_clusters(images, image_paths, clusters, ssim_matrix)
else:
    print("Görüntü bulunamadı veya yüklenemedi.")    