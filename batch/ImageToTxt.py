import base64
import json
import re
import socket
import sys
from urllib.parse import urlparse

import fitz
import pytesseract as tess
import requests
from pdf2image import convert_from_bytes

from batch.SpanishSpellChecker import SpanishSpellChecker
from common.TextractService import TextractService
from common.commonsLib import loggerElk


class ImageToTxt:
    ENCODING = 'utf-8'
    TIMEOUT_SEC = 160
    MIN_TEXT_LENGTH = 10

    def __init__(self, s3service):
        self.logger = loggerElk(__name__)
        self.spell_checker_service = SpanishSpellChecker()
        self.s3_service = s3service
        self.textract = TextractService(self.s3_service)

    def get_ocr_from_pdf2readable(self, s3_service, key):
        self.logger.Information("ImageToTxt::forced native OCR")
        decoded_pages = self.__get_text_by_pdf2readable(
            s3_service=s3_service, bucket=s3_service.real_bucket, key=key,
            force=1, skip=0, force_no_native=0, disable=0)
        sum_native = sum(map(lambda x: len(self.clean(x)), decoded_pages))
        if sum_native < self.MIN_TEXT_LENGTH:
            self.logger.Information(f"ImageToTxt::forced readable OCR")
            decoded_pages = self.__get_text_by_pdf2readable(
                s3_service=s3_service, bucket=s3_service.real_bucket, key=key,
                force=1, skip=sys.maxsize, disable=1)

        return decoded_pages

    def get_ocr_from_pdf2readable_with_cache(self, s3_service, key):
        decoded_pages = self.get_cache_from_pdf2readable(s3_service, key)
        if not decoded_pages:
            self.get_ocr_from_pdf2readable(s3_service, key)

        return decoded_pages

    def get_max_ocr_from_pdf2readable(self, s3_service, key):
        txt_decoded_list = list()
        txt_decoded = self.get_native_from_pdf2readable(s3_service, key)
        txt_decoded_list.append(txt_decoded)
        txt_decoded = self.get_native_ocr_from_pdf2readable(s3_service, key)
        txt_decoded_list.append(txt_decoded)
        txt_decoded = self.get_readable_from_pdf2readable(s3_service, key)
        txt_decoded_list.append(txt_decoded)

        try:
            txt_decoded_length = list(map(lambda x: len(' '.join(x)), txt_decoded_list))
            txt_decoded_max_id = txt_decoded_length.index(max(txt_decoded_length))
            txt_decoded = txt_decoded_list[txt_decoded_max_id]
        except Exception:
            pass

        return txt_decoded

    def get_cache_from_pdf2readable(self, s3_service, key):
        self.logger.Information("ImageToTxt::cached")
        decoded_pages = self.__get_text_by_pdf2readable(
            s3_service=s3_service, bucket=s3_service.real_bucket, key=key,
            force=0, skip=0, force_no_native=0, disable=1)

        return decoded_pages

    def get_native_from_pdf2readable(self, s3_service, key, skip=0):
        self.logger.Information("ImageToTxt::forced native")
        decoded_pages = self.__get_text_by_pdf2readable(
            s3_service=s3_service, bucket=s3_service.real_bucket, key=key,
            force=1, skip=skip, force_no_native=0, disable=1)

        return decoded_pages

    def get_native_ocr_from_pdf2readable(self, s3_service, key):
        self.logger.Information("ImageToTxt::forced native + OCR")
        decoded_pages = self.__get_text_by_pdf2readable(
            s3_service=s3_service, bucket=s3_service.real_bucket, key=key,
            force=1, skip=0, force_no_native=0, disable=0)

        return decoded_pages

    def get_readable_from_pdf2readable(self, s3_service, key):
        self.logger.Information("ImageToTxt::forced readable OCR")
        decoded_pages = self.__get_text_by_pdf2readable(
            s3_service=s3_service, bucket=s3_service.real_bucket, key=key,
            force=1, skip=sys.maxsize, force_no_native=0, disable=1)

        return decoded_pages

    def get_text_by_tesseract(self, doc_image=None):
        full_text = ''

        if doc_image is not None:
            full_text = self.__get_string(doc_image)

        return full_text

    def get_text_by_textract(self, doc_image=None):
        full_text = ''

        if doc_image is not None:
            full_text = self.__get_string(doc_image)

        return full_text

    def __get_string(self, file_input):
        texto = self.__img_to_string(file_input)
        return texto

    @staticmethod
    def __img_to_string(img):
        lang = 'spa+cat'
        config = ('-l ' + lang + ' --oem 1 --psm 3')
        tess.pytesseract.tesseract_cmd = r"tesseract"
        text_string = tess.image_to_string(img, config=config)
        return text_string

    def __get_text_by_pdf2readable(self,
                                   s3_service,
                                   bucket, key,
                                   force=0,
                                   force_no_native=0,
                                   disable=1,
                                   skip=MIN_TEXT_LENGTH):
        pages = list()
        full_text = ''

        if bucket and key:
            full_text = self.__ocr_pdf2readable(
                s3_service, key, force, force_no_native, disable, skip)

        if len(full_text) > 0:
            pages = re.compile(r'\[\[\[\d+\]\]\]').split(full_text)
            if pages:
                pages = [page for page in pages if page]

        return pages

    def __ocr_pdf2readable(self, s3_service, key, force, force_no_native, disable, skip):
        full_text = ''

        headers = {'content-type': 'application/json'}
        payload_ocr = {
            'persistence': 'S3',
            'source': 'S3',
            'bucket': s3_service.bucket,
            'key': key,
            'data': key,
            'lang': 'spa',
            'forcescan': force,
            'disableocr': disable,
            "forcenonative": force_no_native,
            "max_pages": 99,
            "txtLenForSkip": skip,
            "kb_limit_size": 0,
            "queue": ""
        }

        try:
            r_ocr = None
            self.logger.Information(f'ImgToTxt::Real bucket:{s3_service.bucket}')
            self.logger.Information(f'ImgToTxt::Calling ocr_pdf2readable: '
                                    f'{s3_service.readable_url} with: {payload_ocr}')
            if self.ping(s3_service.readable_url):
                r_ocr = requests.post(s3_service.readable_url,
                                      data=json.dumps(payload_ocr),
                                      headers=headers,
                                      timeout=self.TIMEOUT_SEC)
        except Exception as e:
            if str(e) == "('Connection aborted.', RemoteDisconnected(" \
                         "'Remote end closed connection without response'))":
                self.logger.Error(f'ImgToTxt::Exception {e} after {self.TIMEOUT_SEC} seconds.')
            else:
                self.logger.Error(f"ImgToTxt::{e}")
            return ''

        if r_ocr is None or len(r_ocr.text) == 0:
            self.logger.Information(f'ImgToTxt::Error when calling pdf2readable')
            return ''

        try:
            try:
                json_ocr = json.loads(r_ocr.text)
                if not json_ocr or not len(json_ocr) or not isinstance(json_ocr, dict):
                    raise ValueError("JSON result malformed")
            except Exception as _:
                self.logger.Error('ImgToTxt::JSON format error')
                return ''

            if ('status' in json_ocr and json_ocr['status'] == 'False') or \
                    ('statusCode' in json_ocr and json_ocr['statusCode'] == 500):
                self.logger.Error('ImgToTxt::status False or 500')
                return ''
            rt_len = 0
            if json_ocr and 'resultText' in json_ocr and json_ocr['resultText']:
                rt_len = len(json_ocr['resultText'])
            rtn_len = 0
            if json_ocr and 'resultTextNonNative' in json_ocr and json_ocr['resultTextNonNative']:
                rtn_len = len(json_ocr['resultTextNonNative'])
            if rt_len or rtn_len:
                full_text += json_ocr['resultText'] if rt_len >= rtn_len else json_ocr['resultTextNonNative']
        except Exception as ex:
            self.logger.Error(f"ImgToTxt::ocr_pdf2readable service response -> [{str(ex)}]")
            return ''

        decoded_text = self.load_document(full_text)
        if len(decoded_text.strip()) > self.MIN_TEXT_LENGTH:
            self.logger.Information(f'ImgToTxt::ocr_pdf2readable got some text -> '
                                    f'{self.clean(decoded_text)[:self.MIN_TEXT_LENGTH]}')
        elif 0 < len(decoded_text.strip()) <= self.MIN_TEXT_LENGTH:
            self.logger.Information(f'ImgToTxt::ocr_pdf2readable got little text -> '
                                    f'{len(decoded_text.strip())} characters')
            return ''
        else:
            self.logger.Error(f'ImgToTxt::Error when calling pdf2readable. No text returned. '
                              f'Check that the bucket and credentials '
                              f'are in the pdf2readable config')
            return ''

        return decoded_text

    @staticmethod
    def clean(data):
        if data is None:
            return ''

        if type(data) == int or type(data) == float:
            return data

        data = data.strip()
        data = data.replace('\r', '')
        data = data.replace('\n', '')
        data = data.replace('\t', '')

        data = data.replace('.', '')
        data = data.replace(':', '')
        data = data.replace(';', '')
        data = data.replace('\'', '')
        data = data.lower()

        return data

    @staticmethod
    def check_doc_format(key, file_data, txt_decoded):
        try:
            fitz.Document(filetype='pdf', stream=file_data)
            convert_from_bytes(file_data, dpi=1)
        except Exception:
            raise ValueError(key)

        return key, file_data, txt_decoded

    @staticmethod
    def ping(srv_url):
        address = urlparse(srv_url)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            sock.connect((address.hostname, address.port))
        except Exception:
            return False
        else:
            sock.close()
            return True

    @staticmethod
    def load_document(content):
        try:
            full_text = content.decode(ImageToTxt.ENCODING, errors='replace')
        except (UnicodeDecodeError, AttributeError):
            full_text = content.encode(ImageToTxt.ENCODING, errors="replace").decode(
                ImageToTxt.ENCODING, errors='replace')
        try:
            data_encoded = full_text.encode(ImageToTxt.ENCODING, errors="replace")
            b64_encoded = base64.b64encode(data_encoded)
            b64_decoded = base64.b64decode(b64_encoded)
            full_text = b64_decoded.decode(ImageToTxt.ENCODING, errors='replace')
            try:
                if base64.b64encode(base64.b64decode(data_encoded)) == data_encoded:
                    b64_decoded = base64.b64decode(b64_decoded)
                    full_text = b64_decoded.decode(ImageToTxt.ENCODING, errors='replace')
            except Exception as _:
                pass
        except Exception as _:
            pass

        return full_text
