import cv2

def edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def grayscale_quantization(frame, levels=4):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    quantized = (gray // (256 // levels)) * (256 // levels)
    return quantized

def contrast_enhancement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def soft_appearance(frame):
    return cv2.GaussianBlur(frame, (7, 7), 0)

def cartoon_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=250, sigmaSpace=250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon
