import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import time

INPUT_FOLDER = "test-data"
OUTPUT_FOLDER = "test-output_rotated11"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def rotate_image(image, angle):
    h, w = image.shape
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def extract_brain_region(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image[y:y+h, x:x+w], (x, y, w, h)

def compute_ssim_with_map(left, right):
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left = cv2.resize(left, (w, h))
    right = cv2.resize(right, (w, h))
    right_flipped = cv2.flip(right, 1)
    score, _ = ssim(left, right_flipped, full=True)
    return score

def find_best_angle(gray_resized, coarse_step=10, fine_range=5, min_angle=-50, max_angle=50, min_slices=10, max_slices=30):
    def evaluate_angles(angles):
        best_score = -1
        best_angle = 0
        best_x = -1

        for angle in angles:
            rotated = rotate_image(gray_resized, angle)
            brain_region, _ = extract_brain_region(rotated) if extract_brain_region(rotated) else (None, None)
            if brain_region is None:
                continue

            h, w = brain_region.shape
            local_best_score = -1
            local_best_x = -1

            split_range = np.linspace(int(w * 0.05), int(w * 0.95), num=40, dtype=int)
            for x in split_range:
                left = brain_region[:, :x]
                right = brain_region[:, x:]
                if left.shape[1] < 10 or right.shape[1] < 10:
                    continue

                score = compute_ssim_with_map(left, right)
                if score > local_best_score:
                    local_best_score = score
                    local_best_x = x

            if local_best_score > best_score:
                best_score = local_best_score
                best_angle = angle
                best_x = local_best_x

        return best_angle, best_score, best_x

    coarse_angles = list(range(min_angle, max_angle + 1, coarse_step))
    best_coarse_angle, _, _ = evaluate_angles(coarse_angles)

    fine_angles = list(range(best_coarse_angle - fine_range, best_coarse_angle + fine_range + 1))
    return evaluate_angles(fine_angles)

def process_file(filename):
    path = os.path.join(INPUT_FOLDER, filename)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    name = os.path.splitext(filename)[0]
    gray_resized = cv2.resize(gray, (256, 256))
    best_angle, best_score, best_split_x = find_best_angle(gray_resized)

    rotated_full = rotate_image(gray, best_angle)
    out_path = os.path.join(OUTPUT_FOLDER, f"{name}.png")
    cv2.imwrite(out_path, rotated_full)

    return (name, best_angle, best_score, best_split_x)

def main():
    start = time.time()
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"[🧠] Toplam {len(files)} dosya işlenecek...")

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="İşleniyor"))

    results = [r for r in results if r is not None]
    duration = time.time() - start
    print(f"\n✅ Tüm işlemler tamamlandı. Toplam süre: {duration:.2f} saniye")

if __name__ == "__main__":
    main()
