# OpenCV Image & Video Processing Project
made by: **E. Mohammad Bashar Khalaf**

This project is part of an academic assignment focused on digital image and video processing using OpenCV and Python. The project is divided into two main parts:

---

## 📸 Part 1: Image Processing Pipeline

- Load a real-world image captured by a camera.
- Add a visible watermark (with name and student ID) to a random position.
- Convert the image to grayscale.
- Display image dimensions and pixel value statistics (min, max, mean).
- Apply brightness transformation using the formula `s = c * r`, where `c` is a random value between 0.4 and 2.0.
- Analyze brightness by plotting the histogram before and after the transformation.
- Correct brightness using histogram equalization.
- Add salt-and-pepper noise to the image.
- Denoise the image using both mean and median filters.
- Apply a sharpening filter to correct any post-processing artifacts.
- Save all intermediate and final processed images in `images/processed/`.

---

## 🎥 Part 2: Real-Time Video Processing

- Open the device’s camera and apply real-time filters.
- Supported display modes (you can switch between them during execution):
  1. **Edge Detection**
  2. **Grayscale Quantization**
  3. **Histogram-based Contrast Enhancement**
  4. **Soft and Polished Appearance (Blurring)**
  5. **Cartoon-style Filter**

---

## ✅ Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

All dependencies are listed in the `requirements.txt` file.

write this command in vs code terminal:
 **pip install -r requirements.txt
---

## 💻 Installation Instructions

Follow these steps to run the project on your machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/opencv-project.git
   cd opencv-project
