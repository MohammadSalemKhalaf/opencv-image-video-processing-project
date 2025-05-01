import cv2
from video_filters import *

def main():
    print("Real-time Filter Switching")
    print("Press keys to switch filters:")
    print("[E] Edge  [G] Grayscale Quant.  [C] Contrast")
    print("[S] Soft Appearance  [T] Cartoon  [Q] Quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access the camera.")
        return

    current_filter = "none"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply current filter
        if current_filter == "edge":
            output = edge_detection(frame)
        elif current_filter == "gray":
            output = grayscale_quantization(frame, levels=4)
        elif current_filter == "contrast":
            output = contrast_enhancement(frame)
        elif current_filter == "soft":
            output = soft_appearance(frame)
        elif current_filter == "cartoon":
            output = cartoon_filter(frame)
        else:
            output = frame

        cv2.imshow("Video Filter", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            current_filter = "edge"
        elif key == ord('g'):
            current_filter = "gray"
        elif key == ord('c'):
            current_filter = "contrast"
        elif key == ord('s'):
            current_filter = "soft"
        elif key == ord('t'):
            current_filter = "cartoon"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
