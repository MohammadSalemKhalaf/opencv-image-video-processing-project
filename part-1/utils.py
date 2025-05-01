import cv2
import numpy as np

def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    total_pixels = image.size
    num_salt = int(amount * total_pixels / 2)
    num_pepper = int(amount * total_pixels / 2)

    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

def apply_mean_filter(image):
    return cv2.blur(image, (3, 3))

def apply_median_filter(image):
    return cv2.medianBlur(image, 3)

def correct_brightness(image):
    return cv2.equalizeHist(image)
