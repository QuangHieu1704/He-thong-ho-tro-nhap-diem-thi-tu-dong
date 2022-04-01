from recognition import load_model, doc_bang_diem, recognize_lopthi, recognize_folder

if __name__ == "__main__":
    model = load_model()
    recognize_lopthi(model)
    recognize_folder(model)
    

    


