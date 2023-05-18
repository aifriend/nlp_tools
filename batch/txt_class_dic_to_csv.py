import os

from common.ClassFile import ClassFile
from preprocess.lib.NlpTool import NlpTool
from preprocess.lib.TextPreprocess import TextPreprocess


def load_data():
    file_list = ClassFile.list_files_like(DATA_PATH, "txt")
    file_filter = ClassFile.filter_by_size(file_list)
    print(f"TOTAL FILE: {len(file_filter)}")

    return file_filter


def main(sep=';'):
    nlp_service = NlpTool()
    if os.path.isfile(WORD_SAVE_PATH):
        os.remove(WORD_SAVE_PATH)

    print("Loading data...")
    dataset = load_data()
    left = len(dataset)
    for file in dataset:
        try:
            page_text = ClassFile.get_text(file, encoding=ENCODING)
            full_text = TextPreprocess.load_document(page_text)
        except Exception as _:
            continue

        try:
            class_path, f_name = os.path.split(file)
            root_path, class_id = os.path.split(class_path)
            word_list = nlp_service.filter_word_list(
                full_text,
                stem=False,
                is_spanish=True,
                greater=2,
                token_filter=False,
                vocab_dist=3
            )
            if word_list and word_list:
                data = ' '.join(word_list)[:MAX_LENGTH]
                if data:
                    ClassFile.to_txtfile(
                        data=f"{f_name}{sep}{data}{sep}{class_id}\n",
                        file_=f"{os.path.join(root_path, WORD_SAVE_PATH)}",
                        mode="a+",
                        encoding=ENCODING)
                else:
                    print(f"Enough words to generate text from: {file}")
            else:
                print(f"Words not found or missing: {file}")
        except Exception:
            raise IOError()

        left -= 1
        print(f"Left: {left} / {class_id}")


if __name__ == '__main__':
    ENCODING = 'utf-8'
    MAX_LENGTH = 99999
    DATA_PATH = rf""
    WORD_SAVE_PATH = rf"word_dic_tax_spanish.csv"
    main()
