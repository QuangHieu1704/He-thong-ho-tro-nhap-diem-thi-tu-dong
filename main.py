from recognition import load_model, doc_bang_diem, recognize_file, recognize_folder

if __name__ == "__main__":
    model = load_model()

    file_path = "He ho tro nhap diem tu dong\\Data\\70001_0.png"
    output_path = "He ho tro nhap diem tu dong\\Output"
    recognize_file(model, file_path, output_path)

    # input_path = "He ho tro nhap diem tu dong\\Data\\Bangdiem1"
    # output_path = "He ho tro nhap diem tu dong\\Output"
    # recognize_folder(model, input_path, output_path)
