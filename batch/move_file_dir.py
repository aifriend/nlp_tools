import base64
import os
import random
import re
import shutil

from common.ClassFile import ClassFile


def load_document(content) -> str:
    try:
        full_text = content.decode(ENCODING, errors='replace')
    except (UnicodeDecodeError, AttributeError):
        full_text = content.encode(ENCODING, errors="replace").decode(
            ENCODING, errors='replace')
    try:
        data_encoded = full_text.encode(ENCODING, errors="replace")
        b64_encoded = base64.b64encode(data_encoded)
        b64_decoded = base64.b64decode(b64_encoded)
        full_text = b64_decoded.decode(ENCODING, errors='replace')
        try:
            if base64.b64encode(base64.b64decode(data_encoded)) == data_encoded:
                b64_decoded = base64.b64decode(b64_decoded)
                full_text = b64_decoded.decode(ENCODING, errors='replace')
        except Exception as _:
            pass
    except Exception as _:
        pass

    return full_text


def class_image_to_path(from_image, dst_path, f_name_img):
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    shutil.copy(
        from_image,
        os.path.join(dst_path, f_name_img))


def class_text_to_path(from_path, dst_path, f_name_vocab):
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    shutil.copy(
        os.path.join(from_path, f_name_vocab),
        os.path.join(dst_path, f_name_vocab))


def add_image(class_id: int, img_path, f_path, f_name_img):
    dst_path = None

    if class_id == 0:
        dst_path = os.path.join(f_path, 'other')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 1:
        dst_path = os.path.join(f_path, 'urbana')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 2:
        dst_path = os.path.join(f_path, 'rustica')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 11:
        dst_path = os.path.join(f_path, 'basura')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 14:
        dst_path = os.path.join(f_path, 'residuos')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 15:
        dst_path = os.path.join(f_path, 'alcantarillado')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 23:
        dst_path = os.path.join(f_path, 'saneamiento')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 41:
        dst_path = os.path.join(f_path, 'vado')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 42:
        dst_path = os.path.join(f_path, 'aguas')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 44:
        dst_path = os.path.join(f_path, 'ocupacion')
        class_image_to_path(img_path, dst_path, f_name_img)
    if class_id == 922:
        dst_path = os.path.join(f_path, 'metropolitano')
        class_image_to_path(img_path, dst_path, f_name_img)

    if dst_path is None:
        ClassFile.to_txtfile(
            data=f"{f_name_img}:{class_id}\n",
            file_=os.path.join(f_path, '../', f'_tax_no_class.txt'),
            mode='a+',
            encoding=ENCODING)
    else:
        print(f"Done: {f_name_img} with class {class_id}")


def add_text(img_class_id, f_path, f_name_txt, f_name_vocab):
    dst_path = None

    if img_class_id == 0:
        dst_path = os.path.join(f_path, 'other')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 1:
        dst_path = os.path.join(f_path, 'urbana')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 2:
        dst_path = os.path.join(f_path, 'rustica')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 11:
        dst_path = os.path.join(f_path, 'basura')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 14:
        dst_path = os.path.join(f_path, 'residuos')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 15:
        dst_path = os.path.join(f_path, 'alcantarillado')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 23:
        dst_path = os.path.join(f_path, 'saneamiento')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 41:
        dst_path = os.path.join(f_path, 'vado')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 42:
        dst_path = os.path.join(f_path, 'aguas')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 44:
        dst_path = os.path.join(f_path, 'ocupacion')
        class_text_to_path(f_path, dst_path, f_name_vocab)
    if img_class_id == 922:
        dst_path = os.path.join(f_path, 'metropolitano')
        class_text_to_path(f_path, dst_path, f_name_vocab)

    if dst_path is None:
        ClassFile.to_txtfile(
            data=f"{f_name_txt}:{img_class_id}\n",
            file_=os.path.join(f_path, '../', f'_tax_no_class.txt'),
            mode='a+',
            encoding=ENCODING)
    else:
        print(f"Done: {f_name_txt} with class {img_class_id}")


def db_run():
    doc_file_list = ClassFile.list_files_ext(SOURCE, "jpg")
    for n, doc_img in enumerate(doc_file_list, 1):
        f_path, f_name_img = os.path.split(doc_img)
        f_name_pdf = re.sub(r'\.pdf_\d{2}\.jpg?|\.pdf\.jpg', '.pdf', f_name_img, count=0, flags=0)
        f_name_txt = re.sub(r'\.pdf_\d{2}\.jpg?|\.pdf\.jpg', '.pdf.txt', f_name_img, count=0, flags=0)
        f_name_vocab = f_name_pdf.replace('.pdf', '.pdf.vocab')
        if not os.path.isfile(os.path.join(f_path, f_name_txt)):
            print(f"----------> missing: {f_name_txt}")
            continue
        if not os.path.isfile(os.path.join(f_path, f_name_pdf)):
            print(f"----------> missing: {f_name_pdf}")
            continue
        if not os.path.isfile(os.path.join(f_path, f_name_vocab)):
            print(f"----------> missing: {f_name_vocab}")
            continue
        f_path_dest = os.path.join(f_path, 'move')
        if not os.path.isdir(f_path_dest):
            os.mkdir(f_path_dest)
        shutil.copy(
            os.path.join(f_path, f_name_pdf),
            os.path.join(f_path_dest, f_name_pdf))
        shutil.copy(
            os.path.join(f_path, f_name_img),
            os.path.join(f_path_dest, f_name_img))
        shutil.copy(
            os.path.join(f_path, f_name_txt),
            os.path.join(f_path_dest, f_name_txt))
        shutil.copy(
            os.path.join(f_path, f_name_vocab),
            os.path.join(f_path_dest, f_name_vocab))
        print(f"Done: {f_name_pdf}")


def do_filter():
    doc_file_list = ClassFile.list_files_ext(SOURCE, "txt")
    full_text_list = list()
    for n, doc in enumerate(doc_file_list, 1):
        try:
            content = ClassFile.get_text(doc, encoding=ENCODING)
            full_text = load_document(content)
            if full_text:
                item = (doc, full_text)
                full_text_list.append(item)
        except Exception as _:
            pass

    random.shuffle(full_text_list)
    for p_doc_txt, p_content in full_text_list:
        f_path, f_name_txt = os.path.split(p_doc_txt)
        match_list = list()

        match_list += re.findall(r'SANEAM.|Saneam.|saneam.', p_content)
        match_list += re.findall(r'AGUAS|Aguas|aguas', p_content)
        match_list += re.findall(r'ALCANTARILL|Alcantarill|alcantarill', p_content)
        match_list += re.findall(r'RESIDUOS|Residuos|resuduos', p_content)
        match_list += re.findall(r'BASURA|Basura|basura', p_content)
        match_list += re.findall(r'URBAN.|Urban.|urban.', p_content)
        match_list += re.findall(r'RUSTIC.|Rustic.|rustic.', p_content)
        match_list += re.findall(r'METROPOLI|Metropoli|metropoli', p_content)
        match_list += re.findall(r'\sVADO\s|\svado\s|Vado\s|VEH.CULOS|Veh.culos|veh.culos', p_content)

        if match_list:
            f_name_img = re.sub(r'\.pdf\.txt', '.pdf.jpg', f_name_txt, count=0, flags=0)
            dst_path = os.path.join(f_path, 'filter')
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            shutil.move(
                os.path.join(f_path, f_name_img),
                os.path.join(dst_path, f_name_img))
            shutil.move(
                os.path.join(f_path, f_name_txt),
                os.path.join(dst_path, f_name_txt))
            print(f"Done with {f_name_txt.replace('.txt', '')} with {'#'.join(match_list)}")


if __name__ == '__main__':
    ENCODING = 'utf-8'
    # DB_TAX_NAME = 'tax_img_class'
    SOURCE = r"C:\Users\JoseAntonioFernandez\Documents\DATA\tributos\source\tax_class\KNOWN_TRAIN"
    # db_run()
    do_filter()
