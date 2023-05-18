import base64

from common.ClassFile import ClassFile

if __name__ == '__main__':
    MIN_LENGTH = 50
    path = rf""

    txt_list = ClassFile.list_files_ext(path, "txt")

    total = len(txt_list)
    print(f"TOTAL FILES: {total}\nFROM: {path}")

    left = total
    for txt_key in txt_list:
        txt_content = ClassFile.get_text(txt_key)
        try:
            data_decoded = base64.b64decode(txt_content)
            # ocr_text_decoded = data_decoded.decode('ISO-8859-1', errors='replace')
            ocr_text_decoded = data_decoded.decode('utf-8', errors='replace')
        except Exception as _:
            ocr_text_decoded = txt_content
        if not ocr_text_decoded:
            continue
        # ClassFile.to_txtfile(ocr_text_decoded, txt_key, encoding='ISO-8859-1')
        ClassFile.to_txtfile(ocr_text_decoded, txt_key, encoding='utf-8')
        left -= 1
        print(f"DECODED LEFT: {left}")
