import cv2

def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin
