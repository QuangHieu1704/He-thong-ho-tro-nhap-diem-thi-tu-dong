import re
import cv2
import numpy as np
import os
import pytesseract
import keras_ocr
from PIL import Image


def find_ma_lop(target_img, ref_img):
    target_img = cv2.resize(target_img, (1653, 2337))
    width = target_img.shape[1]
    height = target_img.shape[0]
    result = cv2.matchTemplate(target_img, ref_img, cv2.TM_CCOEFF_NORMED)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    matchLoc = maxLoc
    # cv2.rectangle(target_img, matchLoc, (matchLoc[0] + ref_img.shape[1], matchLoc[1] + ref_img.shape[0]), (0,0,0), 2, 8, 0 )
    # cv2.rectangle(result, matchLoc, (matchLoc[0] + ref_img.shape[1], matchLoc[1] + ref_img.shape[0]), (0,0,0), 2, 8, 0 )
    img_malop = target_img[matchLoc[1] : matchLoc[1] + ref_img.shape[0], matchLoc[0] : matchLoc[0] + ref_img.shape[1]]
    
    return img_malop

if __name__ == "__main__":
    ref_img = cv2.imread("He ho tro nhap diem tu dong\\Data\\ref_image.png", cv2.IMREAD_GRAYSCALE)
    input_folder = "He ho tro nhap diem tu dong\\Data\\Bangdiem1"
    output_folder = "He ho tro nhap diem tu dong\\Output\\Output cut ma lop\\Bang diem 1"
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    for filename in os.listdir(input_folder):
        print("Read ", os.path.join(input_folder, filename))
        target_img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        target_img = cv2.resize(target_img, (1653, 2337), interpolation=cv2.INTER_CUBIC)
        img_malop = find_ma_lop(target_img, ref_img)
        img_malop = cv2.threshold(img_malop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = np.ones((1, 1), np.uint8)
        # img_malop = cv2.dilate(img_malop, kernel, iterations=1)
        img_malop = cv2.erode(img_malop, kernel, iterations=1)  
        img_malop = cv2.resize(img_malop, None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
        img_malop = cv2.GaussianBlur(img_malop, (5, 5), 0)
        img_malop = cv2.bilateralFilter(img_malop, 9, 75, 75)
        text = pytesseract.image_to_string(img_malop, config='--psm 6')
        malop = re.search(r'\d\d\d\d\d\d', text)
        if malop:
            malop = malop.group()
        else:
            malop = ' ERROR '
        save_file = filename.split(".")[0] + " " + str(malop) + ".png"
        print(save_file)
        cv2.imwrite(os.path.join(output_folder, save_file), img_malop)