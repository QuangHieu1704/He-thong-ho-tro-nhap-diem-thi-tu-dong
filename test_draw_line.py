from csv import excel
import cv2
import numpy as np
from process_data import correct_skew, get_lines, image_matching
from PIL import Image
from recognition import load_model, doc_bang_diem, recognize_lopthi, recognize_folder
import re
import pytesseract
import pandas as pd
from keras import backend

characters = u"0123456789.n"
def num_to_label(num):
    label = ""
    for ch in num:
        if ch == -1:
            break
        else:
            label += characters[ch]
    return label


if __name__ == "__main__":
    model = load_model()
    img = Image.open("He ho tro nhap diem tu dong\\Data\\Bangdiem4\\GT3,GT2-CK20191-01.jpg")
    print("type: ", type(img))
    # img = cv2.imread("He ho tro nhap diem tu dong\\Data\\70000_71001\\70001_0.jpg", cv2.IMREAD_COLOR)
    img = np.array(img , dtype=np.uint8)
    print("shape: ", img.shape)
    if img.ndim == 3:
        print("3D")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (1653, 2337), interpolation=cv2.INTER_CUBIC)
    height, width = img.shape
    img = correct_skew(img)
    horizontal_coor, vertical_coor = get_lines(img)
    tmp_img = img
    for y in horizontal_coor:
        cv2.line(tmp_img, (0, y), (width,y), (0, 255, 0), 1)
    for x in vertical_coor:
        cv2.line(tmp_img, (x, 0), (x, height), (0, 255, 0), 1)
    cv2.imshow("img", cv2.resize(tmp_img, None, fx = 0.5, fy = 0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("img", cv2.resize(img, None, fx = 0.5, fy = 0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Đọc mã lớp
    ref_img = cv2.imread("He ho tro nhap diem tu dong\\Data\\ref_image.png", cv2.IMREAD_GRAYSCALE)
    img_malop = image_matching(img, ref_img)
    cv2.imshow("img ma lop", img_malop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_malop = cv2.resize(img_malop, None, fx = 1.5, fy = 1.5, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(img_malop)
    print("ma lop: ", text)
    malop = re.search(r'\d\d\d\d\d\d', text)
    if malop:
        malop = malop.group()
    else:
        malop = 'ERROR '
    excel_filename = str(malop) + '.xlsx'

    # Đọc mã số sinh viên
    list_MSSV = []
    for i in range(1, len(horizontal_coor) - 1):
        img_mssv = img[horizontal_coor[i]: horizontal_coor[i + 1], vertical_coor[1]: vertical_coor[2]]
        cv2.imshow("mssv", img_mssv)
        cv2.waitKey(0)
        img_mssv = cv2.resize(img_mssv, None, fx = 1.5, fy = 1.5, interpolation=cv2.INTER_CUBIC)
        mssv = pytesseract.image_to_string(img_mssv)
        print("mssv: ", mssv)
        mssv = re.search(r'\d\d\d\d\d\d\d\d', mssv)
        if mssv:
            mssv = mssv.group()
        else:
            mssv = pytesseract.image_to_string(img_mssv)
        list_MSSV.append(mssv)
    
    # Đọc điểm
    list_Diem = []
    list_Accuracy_Diem = []
    for i in range(1, len(horizontal_coor) - 1):
        img_diem = img[horizontal_coor[i]: horizontal_coor[i + 1] + round(height * 0.004),
                   vertical_coor[4]: vertical_coor[5]]
        img_diem = cv2.resize(img_diem, (100, 40))
        img_diem = img_diem / 255.0
        pred = model.predict(img_diem.reshape(1, 40, 100, 1))
        decoded = backend.get_value(
            backend.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])
        diem = num_to_label(decoded[0])
        diem = diem.replace("n", "")
        print("diem: ", diem)
        list_Diem.append(diem)

        pred = pred.reshape(20, 13)
        pred = np.max(pred, axis = 1)
        accuracy = np.prod(pred)
        list_Accuracy_Diem.append(accuracy)
    
    df = pd.DataFrame(columns=['MSSV', 'Điểm', 'Accuracy'])
    list_MSSV = pd.Series(list_MSSV)
    list_Diem = pd.Series(list_Diem)
    list_Accuracy_Diem = pd.Series(list_Accuracy_Diem)
    df['MSSV'] = list_MSSV.values
    df['Điểm'] = list_Diem.values
    df['Accuracy'] = list_Accuracy_Diem.values

    print('MA LOP:', excel_filename)
    print(list(list_MSSV))
    print(list(list_Diem))
    # print(df)