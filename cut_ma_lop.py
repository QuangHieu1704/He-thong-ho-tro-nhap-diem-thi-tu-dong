import re
import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR"

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
    ref_img = cv2.imread("Data\\test\\1-1.png", cv2.IMREAD_GRAYSCALE)
    input_folder = "Data\\Bangdiem1"
    output_folder = "Output\\Output cut ma lop"
    for filename in os.listdir(input_folder):
        print("Read ", os.path.join(input_folder, filename))
        target_img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        img_malop = find_ma_lop(target_img, ref_img)

        text = pytesseract.image_to_string(img_malop)
        malop = re.search(r'\d\d\d\d\d\d', text)
        if malop:
            malop = malop.group()
        else:
            malop = ' ERROR ' + filename
        save_file = filename.split(".")[0] + " " + str(malop) + ".png"
        print(save_file)
        cv2.imshow("result", img_malop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite(os.path.join(output_folder, save_file), img_malop)
    
