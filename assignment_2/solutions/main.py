import cv2
import numpy as np


def padding(image, border_width):
    padding_reflect = cv2.copyMakeBorder(
        image, border_width, border_width, border_width, border_width,
                                         cv2.BORDER_REFLECT)
    cv2.imwrite("saved_pictures/padded_lena.png", padding_reflect)
    return padding_reflect


def crop(image):
    height, width, _ = image.shape
    x_0, y_0= 80, 80
    x_1, y_1= width - 130, height -130
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("saved_pictures/cropped_lena.png", cropped_image)
    return cropped_image

def resize(image):
    resized_image = cv2.resize(image, (200,200), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("saved_pictures/resized_lena.png", resized_image)
    return resized_image

def copy(image):
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite("saved_pictures/empty.png", emptyPictureArray)
    return emptyPictureArray

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("saved_pictures/grayscale_lena.png", gray)
    return gray

def hsv(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("saved_pictures/hsv_lena.png", hsvImage)
    return hsvImage

def hue_shifted(image):
    shifted = (image.astype(np.int16) + 50) % 256
    shifted = shifted.astype(np.uint8)
    cv2.imwrite("saved_pictures/hue_shifted_lena.png", shifted)
    return shifted

def smoothing(image):
    gaussian = cv2.GaussianBlur(image, (15, 15), 0)
    cv2.imwrite("saved_pictures/smoothing_lena.png", gaussian)
    return gaussian


def rotation(image):
    angle = input("Enter rotation angle (90 or 180): ")

    if angle == "90":
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == "180":
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        print("Invalid input. Please type 90 or 180.")
        return None

    cv2.imwrite("saved_pictures/rotated_lena.png", rotated_image)
    return rotated_image


def main():

    image = cv2.imread('lena.png')

    operations = {
        "1": lambda: padding(image, 100),
        "2": lambda: crop(image),
        "3": lambda: resize(image),
        "4": lambda: copy(image),
        "5": lambda: grayscale(image),
        "6": lambda: hsv(image),
        "7": lambda: hue_shifted(image),
        "8": lambda: smoothing(image),
        "9": lambda: rotation(image)
    }

    while True:
        print("\nChoose an operation:")
        print("1 - Padding")
        print("2 - Cropping")
        print("3 - Resizing")
        print("4 - Copying")
        print("5 - Grayscale")
        print("6 - HSV")
        print("7 - Hue shift")
        print("8 - Smoothing")
        print("9 - Rotation")
        print("0 - Exit")

        choice = input("Enter your choice: ")

        if choice == "0":
            break
        elif choice in operations:
            result = operations[choice]()
            if result is not None:
                cv2.imshow("Result", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Invalid choice, try again.")


if __name__ == '__main__':
    main()
