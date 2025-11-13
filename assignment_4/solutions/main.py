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


def feature_matching_sift(image_to_align, reference_image, max_features, good_match_precent):
    MIN_MATCH_COUNT = 10

    sift = cv2.SIFT_create(nfeatures=max_features)

    kp1, des1 = sift.detectAndCompute(image_to_align, None)
    kp2, des2 = sift.detectAndCompute(reference_image, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    aligned=None

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h_ref, w_ref = reference_image.shape[:2]
        aligned = cv2.warpPerspective(image_to_align, M, (w_ref, h_ref))

        h, w = image_to_align.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        reference_with_box = reference_image.copy()
        reference_with_box = cv2.polylines(reference_with_box, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    matches = cv2.drawMatches(image_to_align, kp1, reference_image, kp2, good, None, **draw_params)

    cv2.imwrite('saved_pictures/matches.jpg', matches)
    if aligned is not None:
        cv2.imwrite('./saved_pictures/aligned.jpg', aligned)

    plt.imshow(matches, 'gray')
    plt.show()



    return aligned, matches


def resize(image, scale_factor=2, up_or_down="up"):
    pass

def main():
    image_1 = cv2.imread('./reference_images/2_1/reference_img.png')
    image_2 = cv2.imread('./reference_images/2_3/align_this.jpg', cv2.IMREAD_GRAYSCALE)
    image_3 = cv2.imread('./reference_images/2_3/reference_img-1.png', cv2.IMREAD_GRAYSCALE)

    operations = {
        "1": lambda: harris_corner_detection(image_1),
        "2": lambda: feature_matching_sift(image_2, image_3, max_features=1000, good_match_precent=0.7),
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
            result = operations[choice]()
            if result is not None:
                if isinstance(result, tuple):
                    aligned, matches = result
                    if matches is not None:
                        cv2.imshow("Matches", matches)
                    if aligned is not None:
                        cv2.imshow("Aligned", aligned)
                else:
                    cv2.imshow("Result", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Invalid choice, try again.")



if __name__ == '__main__':
    main()
