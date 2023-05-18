import io
import time

import boto3
from pdf2image import convert_from_bytes
from textractprettyprinter.t_pretty_print import get_string, Textract_Pretty_Print, Pretty_Print_Table_Format

from common.commonsLib import loggerElk


class TextractService:

    def __init__(self, s3_service):
        self.s3_service = s3_service
        if s3_service is not None:
            self.client = boto3.client('textract',
                                       endpoint_url=f'https://textract.{self.s3_service.region}.amazonaws.com',
                                       region_name=self.s3_service.region,
                                       aws_access_key_id=self.s3_service.aws_access_key_id,
                                       aws_secret_access_key=self.s3_service.aws_secret_access_key)
        self.logger = loggerElk(__name__)

    def startJob(self, s3_bucket_name, object_name):
        _response = None

        _response = self.client.start_document_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': s3_bucket_name,
                    'Name': object_name
                }
            },
            FeatureTypes=["TABLES", "FORMS"]
        )

        return _response["JobId"]

    def isJobComplete(self, job_id):
        time.sleep(5)

        _response = self.client.get_document_analysis(JobId=job_id)
        status = _response["JobStatus"]
        self.logger.Information("TextractService::Job status {}".format(status))

        while status == "IN_PROGRESS":
            time.sleep(5)
            _response = self.client.get_document_analysis(JobId=job_id)
            status = _response["JobStatus"]
            self.logger.Information("TextractService::Job status {}".format(status))

        return status

    def getJobResults(self, job_id):
        pages = []

        time.sleep(5)

        _response = self.client.get_document_analysis(JobId=job_id)

        pages.append(_response)
        self.logger.Information("TextractService::Resultset recieved page {}"
                                .format(len(pages)))
        next_token = None
        if 'NextToken' in _response:
            next_token = _response['NextToken']

        while next_token:
            time.sleep(5)

            _response = self.client.get_document_analysis(JobId=job_id, NextToken=next_token)

            pages.append(_response)
            self.logger.Information("TextractService::Resultset recieved page {}"
                                    .format(len(pages)))
            next_token = None
            if 'NextToken' in _response:
                next_token = _response['NextToken']

        return pages

    def processPdf(self, key):
        response = None
        self.logger.Information(f"TextractService::processPdf: bucket: {self.s3_service.real_bucket}")
        self.logger.Information(f"TextractService::processPdf: key: {key}")
        job_id = self.startJob(self.s3_service.real_bucket, key)
        self.logger.Information("TextractService::Started job with id {}".format(job_id))
        if self.isJobComplete(job_id):
            response = self.getJobResults(job_id)

        return response

    def processImage(self, key):
        try:
            self.logger.Information(f"TextractService::processImage: bucket: {self.s3_service.real_bucket}")
            self.logger.Information(f"TextractService::processImage: key: {key}")
            doc_content = self.client.analyze_document(
                Document={
                    'S3Object': {
                        'Bucket': self.s3_service.real_bucket,
                        'Name': key
                    }
                },
                FeatureTypes=['TABLES', 'FORMS'])
        except Exception:
            doc_content = ''

        return doc_content

    def processPdftoImage(self, key, data_decoded, page_number=1):
        images = convert_from_bytes(data_decoded)
        img_key = key + '.img.1'

        img_byte_arr = io.BytesIO()
        images[page_number - 1].convert('L').save(img_byte_arr, format='PNG')

        self.s3_service.upload_file(img_key, img_byte_arr.getvalue())

        self.logger.Information(f"TextractService::processPNG: bucket: {self.s3_service.real_bucket}")
        self.logger.Information(f"TextractService::processPNG: key: {img_key}")
        try:
            doc_content = self.client.analyze_document(
                Document={
                    'S3Object': {
                        'Bucket': self.s3_service.real_bucket,
                        'Name': img_key
                    }
                },
                FeatureTypes=['TABLES', 'FORMS'])
        except Exception:
            doc_content = ''

        return doc_content

    @staticmethod
    def textract_response_formatter(response):
        # Detect columns and print lines
        columns = []
        lines = []
        for i in range(len(response)):
            if "Blocks" in response[i]:
                for item in response[i]["Blocks"]:
                    if item["BlockType"] == "LINE":
                        column_found = False
                        for index, column in enumerate(columns):
                            bbox_left = item["Geometry"]["BoundingBox"]["Left"]
                            bbox_right = item["Geometry"]["BoundingBox"]["Left"] + item["Geometry"]["BoundingBox"][
                                "Width"]
                            bbox_centre = item["Geometry"]["BoundingBox"]["Left"] + item["Geometry"]["BoundingBox"][
                                "Width"] / 2
                            column_centre = column['left'] + column['right'] / 2

                            if (column['left'] < bbox_centre < column['right']) or (
                                    bbox_left < column_centre < bbox_right):
                                # Bbox appears inside the column
                                lines.append([index, item["Text"]])
                                column_found = True
                                break
                        if not column_found:
                            columns.append({'left': item["Geometry"]["BoundingBox"]["Left"],
                                            'right': item["Geometry"]["BoundingBox"]["Left"] +
                                                     item["Geometry"]["BoundingBox"]["Width"]})
                            lines.append([len(columns) - 1, item["Text"]])

        pretty_printed_string = get_string(textract_json=response,
                                           output_type=[
                                               Textract_Pretty_Print.LINES,
                                               Textract_Pretty_Print.TABLES,
                                               Textract_Pretty_Print.FORMS,
                                               # Textract_Pretty_Print.WORDS,
                                           ],
                                           table_format=Pretty_Print_Table_Format.plain)

        return lines, pretty_printed_string
