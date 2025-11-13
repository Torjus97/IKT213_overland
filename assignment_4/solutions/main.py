import cv2
import numpy as np
from matplotlib import pyplot as plt


def harris_corner_detection(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)
    reference_image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imwrite('./saved_pictures/Harris_Corner_Detection.jpg', reference_image)
    return reference_image


def feature_matching_sift(image_to_align, reference_image, max_features, good_match_percent):
    MIN_MATCH_COUNT = 10

    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_features)

    # Convert to grayscale if needed
    img1_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY) if len(image_to_align.shape) == 3 else image_to_align
    img2_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY) if len(
        reference_image.shape) == 3 else reference_image

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)

    print(f"Found {len(good)} good matches")

    if len(good) > MIN_MATCH_COUNT:
        # Find homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Warp image to align
        h, w = img2_gray.shape
        aligned = cv2.warpPerspective(img1_gray, M, (w, h))

        # Draw matches
        draw_params = dict(matchColor=(255, 0, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        matches_img = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, good, None, **draw_params)

        # Save images
        cv2.imwrite('./saved_pictures/matches.jpg', matches_img)
        cv2.imwrite('./saved_pictures/aligned.jpg', aligned)

        # Display
        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(matches_img, cmap='gray')
        plt.title('Matches')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(aligned, cmap='gray')
        plt.title('Aligned')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        return aligned, matches_img
    else:
        print(f"Not enough matches found: {len(good)}/{MIN_MATCH_COUNT}")
        return None, None


def main():
    image_1 = cv2.imread('./reference_images/2_1/reference_img.png')
    image_2 = cv2.imread('./reference_images/2_3/align_this.jpg')
    image_3 = cv2.imread('./reference_images/2_3/reference_img-1.png')

    operations = {
        "1": lambda: harris_corner_detection(image_1),
        "2": lambda: feature_matching_sift(image_2, image_3, max_features=10, good_match_percent=0.7),
    }

    while True:
        print("\nChoose an operation:")
        print("1 - Key feature matching using Harris Corner Detection.")
        print("2 - Feature based image alignment using SIFT.")
        print("0 - Exit")

        choice = input("Enter your choice: ")

        if choice == "0":
            break
        elif choice in operations:
            operations[choice]()
        else:
            print("Invalid choice, try again.")


if __name__ == '__main__':
    main()