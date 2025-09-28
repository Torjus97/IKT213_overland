import cv2
import numpy as np
from matplotlib import pyplot as plt

def sobel_edge_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)

    sobelxy = (255*cv2.Sobel(image_blur,cv2.CV_64F,1,1, ksize=1) ).clip(0,255).astype(np.uint8)
    cv2.imwrite("saved_pictures/sobel_lambo.png", sobelxy)
    return sobelxy


def canny_edge_detection(image):
    image_blur = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(image_blur, 50, 50)

    plt.imshow(edges, cmap='gray')
    plt.xticks([]), plt.yticks([])

    cv2.imwrite("saved_pictures/canny_lambo.png", edges)
    return edges


def template_match(image, template):

    image_rgb = image
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite('saved_pictures/template_match.png', image_rgb)

    return image_rgb

def resize(image, scale_factor=2, up_or_down="up"):
    rows, cols, _ = image.shape

    if up_or_down == "up":
        resized = cv2.pyrUp(image, dstsize=(cols * scale_factor, rows * scale_factor))
        filename = f"saved_pictures/resized_up_{scale_factor}x.png"
        print(f"Zoomed in: {scale_factor}x")

    elif up_or_down == "down":
        resized = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))
        filename = f"saved_pictures/resized_down_{scale_factor}x.png"
        print(f"Zoomed out: /{scale_factor}")

    else:
        print("Invalid up_or_down argument, must be 'up' or 'down'")
        return None

    cv2.imwrite(filename, resized)
    print(f"Saved resized image as {filename}")
    return resized

def main():
    image_1 = cv2.imread('lambo.png')
    image_2 = cv2.imread('shapes-1.png')
    image_3 = cv2.imread('shapes_template.jpg')

    operations = {
        "1": lambda: sobel_edge_detection(image_1),
        "2": lambda: canny_edge_detection(image_1),
        "3": lambda: template_match(image_2, image_3),
        "4": lambda:  resize(
            image_1,
            scale_factor=int(input("Enter scale factor (default 2): ") or 2),
            up_or_down=input("Enter 'up' to zoom in or 'down' to zoom out: ").strip().lower()
        )
    }


    while True:
        print("\nChoose an operation:")
        print("1 - Sobel Edge Detection")
        print("2 - Canny Edge Detection")
        print("3 - Template Match")
        print("4 - Resizing")
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
