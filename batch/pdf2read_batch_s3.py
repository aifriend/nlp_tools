import base64
import os

from batch.PDFImageService import PDFImageService

os.environ["LIBRARIES_LOG_LEVEL"] = "ERROR"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["OCR_IMAGE_API_URL"] = "http://192.168.1.1:7000/api/ocr/image"

from batch.Configuration import Configuration
from batch.ImageToTxt import ImageToTxt
from batch.S3Service import S3Service
from common.ClassFile import ClassFile


def _request_pdf_to_readable(_path, _bucket, _domain):
    conf = Configuration()
    s3_service = S3Service(
        config=conf.load_aws_config(), bucket=_bucket, domain=_domain)
    ocr_service = ImageToTxt(s3_service)
    key_list = ClassFile.list_pdf_files(_path)
    total = len(key_list)
    total_doc = total
    for key in key_list:
        f_path, f_name = os.path.split(key)
        if True or f_name == "":
            try:
                if os.stat(key).st_size > 0:
                    f_path, f_name = os.path.split(key)
                    print(f"Doc: {os.path.join(f_path, f'{f_name}.txt')}")
                    if os.path.isfile(os.path.join(f_path, f"{f_name}.txt")):
                        print(f"From cache key: {f_name}")
                        total_doc -= 1
                        continue

                    name = f_name
                    if s3_service.domain:
                        name = f"{s3_service.domain}/{f_name}"

                    decoded_pages = ocr_service.get_ocr_from_pdf2readable(s3_service, name)
                    sum_page_char = sum(map(lambda x: len(x), decoded_pages))
                    if not sum_page_char:
                        page_list = list()
                        pdf_image_service = PDFImageService()
                        pages = pdf_image_service.load_pages(key)
                        if len(pages):
                            print(f"Run tesseract for {len(pages)} pages: ", end='')
                            for n, page in enumerate(pages):
                                page_list.append(ocr_service.get_text_by_tesseract(doc_image=page))
                                print(f"{n + 1}#", end='')
                            print()
                            sum_page_char = sum(map(lambda x: len(x), page_list))
                            if sum_page_char:
                                doc_content = "".join(page_list)
                                # plain text
                                ClassFile.to_txtfile(doc_content,
                                                     os.path.join(f_path, f"{f_name}.txt"),
                                                     encoding=ENCODING)
                                # b64 encoded text
                                b_text = doc_content.encode(ENCODING, 'replace')
                                decode = b_text.decode(ENCODING, 'replace')
                                base64_bytes = base64.b64encode(bytes(decode, ENCODING))
                                ClassFile.to_txtfile(base64_bytes,
                                                     file_=os.path.join(f_path, f"{f_name}.b64.txt"),
                                                     mode="wb",
                                                     encoding=ENCODING)
                    print(f"OCR size: {sum_page_char}")
                else:
                    print(f"Done: {key}")

            except Exception as e:
                print(f"ERROR with {key}: {e}\n")

            total_doc -= 1
            print(f"LEFT: {total_doc}/{total}")


if __name__ == '__main__':
    ENCODING = 'utf-8'
    domain = ""
    bucket = ""
    path = os.path.join(rf"")
    _request_pdf_to_readable(path, bucket, domain)
