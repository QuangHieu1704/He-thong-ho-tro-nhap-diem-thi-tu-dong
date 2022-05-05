import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    target_img = cv2.imread("Data\\test\\24-2.png", cv2.IMREAD_GRAYSCALE)
    width = target_img.shape[1]
    height = target_img.shape[0]
    ref_img = cv2.imread("Data\\test\\1-1.png", cv2.IMREAD_GRAYSCALE)

    result = cv2.matchTemplate(target_img, ref_img, cv2.TM_CCOEFF_NORMED)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    matchLoc = maxLoc
    cv2.rectangle(target_img, matchLoc, (matchLoc[0] + ref_img.shape[1], matchLoc[1] + ref_img.shape[0]), (0,0,0), 2, 8, 0 )
    cv2.rectangle(result, matchLoc, (matchLoc[0] + ref_img.shape[1], matchLoc[1] + ref_img.shape[0]), (0,0,0), 2, 8, 0 )
    cv2.imshow("target ", target_img)
    cv2.imshow("result ", result)
    
    # Show the final image with the matched area.
    cv2.waitKey(0)
    cv2.destroyAllWindows(0)
    # sift = cv2.SIFT_create()
    # keypoints_1, descriptors_1 = sift.detectAndCompute(target_img, None)
    # keypoints_2, descriptors_2 = sift.detectAndCompute(ref_img, None)

    # print(len(keypoints_1), len(keypoints_2))

    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)

    # matches = bf.match(descriptors_1,descriptors_2)
    # matches = sorted(matches, key = lambda x:x.distance)

    # kpt = []
    # # for i in range(len(keypoints_2)):
    # #     if 200 <= keypoints_2[i].pt[0] <= 420 and  340 <= keypoints_2[i].pt[0] <= 400:
    # #         print(i)
    # #         kpt.append(list(keypoints_1[i].pt))
    # for i in range(len(keypoints_2)):
    #         kpt.append(list(keypoints_1[i].pt))
    # kpt = np.asarray(kpt)
    # kpt = kpt.astype(int)
    # max_point = tuple(kpt.max(axis = 0))
    # max_point = (max_point[1], max_point[0])
    # min_point = tuple(kpt.min(axis = 0))
    # min_point = (min_point[1], min_point[0])
    # print(max_point)
    # print(min_point)
    # img3 = cv2.rectangle(target_img, min_point, max_point, (0, 0, 0), 2)
    # cv2.imwrite("Data\\test\\test_matching_result.png", img3)
