import io
import os

from PyPDF2 import PdfFileWriter, PdfReader


class DocumentFactory:

    @staticmethod
    def get_doc_id_list(doc_list):
        return list(map(lambda x: x.id, doc_list))

    @staticmethod
    def get_doc_tid_list(doc_list):
        return list(map(lambda x: x.tid, doc_list))

    @staticmethod
    def update_doc_number(doc_list) -> None:
        for it, doc in enumerate(doc_list):
            doc.native = doc.number
            doc.number = it

    @staticmethod
    def extract_doc_page_by_type(doc_list, doc_type):
        extract_doc_list = list()
        page_list = doc_list.copy()
        has_doc, start, stop = DocumentFactory.get_doc_by_id(page_list, doc_type, hard=len(doc_list) > 10)
        while has_doc:
            extract_doc_list += page_list[start:stop]
            doc_list_tail = page_list[stop:]
            doc_list_head = page_list[:start]
            page_list = doc_list_head + doc_list_tail
            DocumentFactory.update_doc_number(page_list)
            has_doc, start, stop = DocumentFactory.get_doc_by_id(page_list, doc_type, hard=len(doc_list) > 10)

        return extract_doc_list, page_list

    @staticmethod
    def get_doc_by_id(doc_list, doc_type, hard=False):
        found = False
        start = -1
        stop = len(doc_list)
        if hard:
            for doc in doc_list:
                if doc.id == doc_type:
                    if not found:
                        start = doc.number
                    found = True
                elif found:
                    stop = doc.number
                    break
        else:
            for doc in doc_list:
                if doc.id == doc_type or doc.tid == doc_type:
                    if not found:
                        start = doc.number
                    found = True
                elif found:
                    stop = doc.number
                    break

        return 0 <= start < stop and stop >= 0, start, stop

    @staticmethod
    def doc_page_to_pdf(doc_list):
        doc_byte_arr = None
        if not doc_list:
            return doc_byte_arr

        doc_image_list = list()
        for doc in doc_list:
            doc_image_list.append(doc.image.convert('RGB'))

        doc_byte_arr = io.BytesIO()
        doc_image_list[0].save(
            doc_byte_arr,
            format='PDF',
            save_all=True,
            append_images=doc_image_list[1:]
        )
        doc_byte_arr = doc_byte_arr.getvalue()

        return doc_byte_arr

    @staticmethod
    def image_list_to_bw_pdf(image_list):
        doc_byte_arr = None
        if not image_list:
            return doc_byte_arr

        doc_image_list = list()
        for doc in image_list:
            doc_image_list.append(doc.image.convert('L'))

        doc_byte_arr = io.BytesIO()
        doc_image_list[0].save(
            doc_byte_arr,
            format='PDF',
            save_all=True,
            append_images=doc_image_list[1:]
        )
        doc_byte_arr = doc_byte_arr.getvalue()

        return doc_byte_arr

    @staticmethod
    def doc_content_to_pdf(split_page_list, key, new_name, pdf_image_service):
        if split_page_list:
            output = PdfFileWriter()
            with open(os.path.join(pdf_image_service.f.name, key), "rb") as i_stream:
                input_pdf = PdfReader(i_stream, strict=False)
                for split_page in split_page_list:
                    try:
                        output.add_page(input_pdf.pages[split_page.native_number])
                    except Exception as _:
                        continue

                with open(os.path.join(pdf_image_service.f.name, new_name), "wb") as outputStream:
                    output.write(outputStream)

                return new_name

        return key
