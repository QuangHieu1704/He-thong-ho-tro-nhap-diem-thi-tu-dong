import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

if __name__ == "__main__":
    img = cv2.imread("He ho tro nhap diem tu dong\\Data\\test\\test skew.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("skew", cv2.resize(img, None, fx = 0.5, fy = 0.5))
    cv2.waitKey(0)
    img = correct_skew(img)
    cv2.imshow("skew", cv2.resize(img, None, fx = 0.5, fy = 0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()