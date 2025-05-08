import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *

os.makedirs('images/processed', exist_ok=True)

def add_watermark(image, text, position=(50, 50)):
    watermarked = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    cv2.putText(watermarked, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return watermarked

original_img = cv2.imread('images/input.jpg')
watermarked_img = add_watermark(original_img, 'Mohammad Khalaf - 12112969', position=(50, 50))
watermarked_img = add_watermark(watermarked_img, 'Ayham Fuqha - 12217066', position=(50, 100))
cv2.imwrite('images/watermark.jpg', watermarked_img)

# Convert to grayscale
img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('images/processed/grayscale.jpg', img)

# Statistics
height, width = img.shape
channels = 1
mean_val = np.mean(img)
min_val = np.min(img)
max_val = np.max(img)

print("Image Statistics:")
print(f"Dimensions: {height}x{width}")
print(f"Number of Color Channels: {channels}")
print(f"Mean Pixel Value: {mean_val:.2f}")
print(f"Min Pixel Value: {min_val}")
print(f"Max Pixel Value: {max_val}")

# Brightness modification
c = np.random.uniform(0.4, 2.0)
bright = np.clip(c * img, 0, 255).astype(np.uint8)
cv2.imwrite('images/processed/bright.jpg', bright)

# Brightness correction using histogram equalization
corrected = cv2.equalizeHist(bright)
cv2.imwrite('images/processed/corrected.jpg', corrected)

# Histogram comparison: Grayscale vs Brightness
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(img.ravel(), 256, [0, 256], color='gray')
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(bright.ravel(), 256, [0, 256], color='orange')
plt.title(f"Brightness Adjusted Histogram (c = {c:.2f})")
plt.xlabel("Pixel Intensity")

plt.tight_layout()
plt.savefig('images/processed/compare_grayscale_brightness.png')
plt.close()

# Histogram comparison: Grayscale vs Corrected
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(img.ravel(), 256, [0, 256], color='gray')
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(corrected.ravel(), 256, [0, 256], color='green')
plt.title("Corrected Histogram (Equalized)")
plt.xlabel("Pixel Intensity")

plt.tight_layout()
plt.savefig('images/processed/compare_grayscale_corrected.png')
plt.close()

# Add salt-and-pepper noise
def add_salt_and_pepper_noise(image, amount):
    noisy_image = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    # Salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

noisy = add_salt_and_pepper_noise(corrected, 0.05)
cv2.imwrite('images/processed/noisy.jpg', noisy)

# Apply mean filter
mean_filtered = cv2.blur(noisy, (3, 3))
cv2.imwrite('images/processed/mean_filtered.jpg', mean_filtered)

# Apply median filter
median_filtered = cv2.medianBlur(noisy, 3)
cv2.imwrite('images/processed/median_filtered.jpg', median_filtered)

# Apply sharpening after noise reduction
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(median_filtered, -1, sharpen_kernel)
cv2.imwrite('images/processed/sharpened.jpg', sharpened)
