import io
import os
import tempfile
from subprocess import check_output

from pdf2image import convert_from_bytes, convert_from_path

from batch.DocumentFactory import DocumentFactory
from common.commonsLib import loggerElk


class PDFImageService:
    def __init__(self):
        tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.f = tempfile.TemporaryDirectory(dir=tmp_path)
        self.logger = loggerElk(__name__)

    def save_temporarily(self, data, key):
        if data is None:
            self.logger.Information("PDFImageService::data is None and can't be saved in temporarily directory")
        full_path = os.path.join(self.f.name, key)
        try:
            os.makedirs(os.path.dirname(full_path))
        except Exception:
            pass
        original = open(os.path.join(self.f.name, key), "wb")
        original.write(data)
        original.flush()
        original.close()

    @staticmethod
    def get_first_page(data):
        images = convert_from_bytes(data)
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        return img_byte_arr

    @staticmethod
    def get_pages(data, dpi=200, gray=False):
        page_list = convert_from_bytes(data, dpi=dpi, grayscale=gray)
        return page_list

    @staticmethod
    def load_pages(filename, dpi=200):
        page_list = convert_from_path(filename, dpi=dpi)
        return page_list

    def clean(self):
        if os.path.exists(self.f.name):
            self.f.cleanup()

    def get_num_pages(self, key):
        output = check_output(["pdfinfo", os.path.join(self.f.name, key)]).decode("utf-8", "replace")
        num_pages = int([line for line in output.splitlines() if "Pages:" in line][0].split(":")[1])
        return num_pages

    def doc2Pdf2s3(self, key, image_list, s3_service):
        # doc to pdf
        pdf_name = f"{key}.pdf"
        data_decoded = DocumentFactory.image_list_to_bw_pdf(image_list)
        pdf_serv = PDFImageService()
        pdf_serv.save_temporarily(data_decoded, pdf_name)

        # save cache
        if not s3_service.check_file(pdf_name):
            self.logger.Information(f'PDFImageService::Saved doc name: {pdf_name} to S3')
            s3_service.upload_file_by_path(
                os.path.join(pdf_serv.f.name, pdf_name), pdf_name)

        # clean pdf service
        if 'pdf_serv' in locals() or 'pdf_serv' in globals():
            pdf_serv.clean()

        return pdf_name

    def pil2Pdf(self, images, key):
        full_path = os.path.join(self.f.name, key)
        try:
            os.makedirs(os.path.dirname(full_path))
        except Exception:
            pass
        if len(images) > 1:
            images[0].save(full_path, 'PDF', resolution=100.0, save_all=True, append_images=images[1:])
        else:
            images[0].save(full_path, 'PDF', resolution=100.0, save_all=False)
        return key

    @staticmethod
    def convert_to_L(images):
        new_images = []
        for image in images:
            if image.mode != 'L':
                new_images.append(image.convert('L'))
            else:
                new_images.append(image)
        return new_images
