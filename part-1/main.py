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
watermarked_img = add_watermark(original_img, 'Mohammad Khalaf - 12112969')
cv2.imwrite('images/watermark.jpg', watermarked_img)

img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('images/processed/grayscale.jpg', img)

print("Dimensions:", img.shape)
print("Min pixel:", np.min(img))
print("Max pixel:", np.max(img))
print("Mean pixel:", np.mean(img))

c = np.random.uniform(0.4, 2.0)
bright = np.clip(c * img, 0, 255).astype(np.uint8)
cv2.imwrite('images/processed/bright.jpg', bright)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.hist(img.ravel(), 256), plt.title("Original Histogram")
plt.subplot(1,2,2), plt.hist(bright.ravel(), 256), plt.title("After Brightness Change")
plt.savefig('images/processed/histogram_comparison.png')
plt.close()

corrected = cv2.equalizeHist(bright)
cv2.imwrite('images/processed/corrected.jpg', corrected)

def add_salt_and_pepper_noise(image, amount):
    noisy_image = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

noisy = add_salt_and_pepper_noise(corrected, 0.05)
cv2.imwrite('images/processed/noisy.jpg', noisy)

mean_filtered = cv2.blur(noisy, (3, 3))
median_filtered = cv2.medianBlur(noisy, 3)
cv2.imwrite('images/processed/mean_filtered.jpg', mean_filtered)
cv2.imwrite('images/processed/median_filtered.jpg', median_filtered)

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(median_filtered, -1, sharpen_kernel)
cv2.imwrite('images/processed/sharpened.jpg', sharpened)
