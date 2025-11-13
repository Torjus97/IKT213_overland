import cv2


def print_camera_info(cam):
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cam.get(cv2.CAP_PROP_FPS)
    print("Frame Width:", frame_width)
    print("Frame Height:", frame_height)
    print("FPS:", fps)

    with open("camera_outputs.txt", "w") as f:
        f.write(f"Height: {frame_height}\n")
        f.write(f"Width: {frame_width}\n")
        f.write(f"FPS: {fps}\n")




def print_image_information(image):
    height, width, channels = image.shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size:", image.size)
    print("datasets Type:", image.dtype)


def main():
    image = cv2.imread(
        "lena-1.png")

    cam = cv2.VideoCapture(0)

    print_image_information(image)

    print_camera_info(cam)

if __name__ == '__main__':
    main()
