import os
import random

from common.ClassFile import ClassFile
from preprocess.lib.NlpTool import NlpTool


def load_data():
    file_list = ClassFile.list_files_like(DATA_PATH, "txt")
    file_filter = ClassFile.filter_by_size(file_list)
    random.shuffle(file_filter)
    print(f"Total docs: {len(file_filter)}")

    return file_filter[:MAX_DOC]


def main(sep='#'):
    nlp_service = NlpTool()
    if os.path.isfile(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)

    print("Loading data...")
    dataset = load_data()
    total = len(dataset)
    for n, file in enumerate(dataset, 1):
        try:
            if n >= MAX_DOC:
                break
            class_path, f_name = os.path.split(file)
            _, class_id = os.path.split(class_path)
            full_text = ClassFile.get_text(file, encoding=ENCODING)
            word_list = nlp_service.pre_process(full_text)
            if word_list is not None and word_list:
                data = ' '.join(word_list)[:MAX_LENGTH]
                if data:
                    ClassFile.to_txtfile(
                        data=f"{f_name}#'{data}'\n",
                        file_=CSV_FILE_PATH,
                        mode="a+",
                        encoding=ENCODING)
                else:
                    print(f">>> Enough words to generate text from: {f_name}")
            else:
                print(f">>> Words not found or missing: {f_name}")
        except Exception:
            raise IOError()

        print(f"Done {n}/{total}: {f_name}")


if __name__ == '__main__':
    ENCODING = 'utf-8'
    MAX_LENGTH = 1024
    MAX_DOC = 100
    DATA_PATH = rf"C:\Users\JoseAntonioFernandez\Documents\DATA\simple_note\GEN"
    CSV_FILE_PATH = rf"output\ns_gen.csv"

    main()
