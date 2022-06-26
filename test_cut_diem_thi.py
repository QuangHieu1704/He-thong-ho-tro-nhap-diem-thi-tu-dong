from re import L
import cv2
import numpy as np
import os
from process_data import get_lines, correct_skew

if __name__ == "__main__":
    input_folder = "He ho tro nhap diem tu dong\\Data\\Bangdiem4"
    output_folder = "He ho tro nhap diem tu dong\\Data\\Trainning Data\\temp"
    count = 10534
    for filename in os.listdir(input_folder):
        print("Read filename: ", filename)
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        img = correct_skew(img)
        horizontal_coor, vertical_coor = get_lines(img)
        for i in range(1, len(horizontal_coor) - 1):
            img_diem = img[horizontal_coor[i]: horizontal_coor[i + 1] + round(height * 0.004),
                    vertical_coor[4]: vertical_coor[5]]
            img_diem = cv2.resize(img_diem, (100, 40))
            name = str(count) + ".jpg"
            print("write", name)
            cv2.imwrite(os.path.join(output_folder, name), img_diem)
            count += 1

