import os
import random

from common.ClassFile import ClassFile
from preprocess.lib.TextPreprocess import TextPreprocess


def load_data():
    file_list = ClassFile.list_files_like(DATA_PATH, "txt")
    file_filter = ClassFile.filter_by_size(file_list)
    print(f"TOTAL FILE: {len(file_filter)}")

    return file_filter


def main():
    if os.path.isfile(WORD_SAVE_PATH):
        os.remove(WORD_SAVE_PATH)

    print("Loading data...")
    dataset = load_data()
    random.shuffle(dataset)
    left = len(dataset)
    for file in dataset:
        try:
            page_text = ClassFile.get_text(file, encoding=ENCODING)
            full_text = TextPreprocess.load_document(page_text)
        except Exception as _:
            continue

        try:
            ClassFile.to_txtfile(
                data=f"'{full_text}'#\n",
                file_=WORD_SAVE_PATH,
                mode="w+",
                encoding=ENCODING)
        except Exception as e:
            raise IOError()

        left -= 1
        print(f"Left: {left} / {len(dataset)}")


if __name__ == '__main__':
    ENCODING = 'utf-8'
    MAX_LENGTH = 500
    DATA_PATH = rf""
    WORD_SAVE_PATH = rf""
    DIC_LOAD_PATH = None
    main()
