import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    target_img = cv2.imread("He ho tro nhap diem tu dong\\Data\\test\\1-2.png", cv2.IMREAD_GRAYSCALE)
    target_img = cv2.resize(target_img, (800, 200))
    ref_img = cv2.imread("He ho tro nhap diem tu dong\\Data\\test\\1-2.png", cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.resize(ref_img, (800, 200))
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(target_img,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(ref_img,None)

    print(len(keypoints_1), len(keypoints_2))

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    print(keypoints_2[10].pt)
    img3 = cv2.drawMatches(target_img, keypoints_1, ref_img, keypoints_2, matches[:533], ref_img, flags=2)
    plt.imshow(img3),plt.show()
