import base64
import io
import warnings

import boto3
from botocore.exceptions import ClientError

from common.s3.Exception import S3FileNotFoundException

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


class S3File:
    def __init__(self, s3_element):
        self.Key = s3_element['Key']
        self.Size = s3_element['Size']

    def get_category(self):
        spl = self.Key.split("/")
        if len(spl) >= 2:
            return spl[len(spl) - 2]
        elif len(spl) == 1:
            return spl[0]
        else:
            raise IOError(f"The element with key {self.Key} is not in a folder for category")


class S3Service(object):
    TIMEOUT = 99999

    def __init__(self, config, bucket, domain=""):
        self.bucket = bucket
        self.domain = domain
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.region = None
        self.__s3_session = None

        try:
            match = next(d for d in config['AWS']['BUCKETS'] if d['ID'] == bucket)
            self.aws_access_key_id = match['ACCESS_KEY_ID']
            self.aws_secret_access_key = match['SECRET_ACCESS_KEY']
            real_name = match.get('REAL_NAME', None)
            self.region = match.get('REGION', None)
            if real_name is not None:
                self.bucket = real_name
        except Exception as e:
            match = next(d for d in config['AWS']['BUCKETS'] if d['ID'] == 'DEFAULT')
            self.aws_access_key_id = match['ACCESS_KEY_ID']
            self.aws_secret_access_key = match['SECRET_ACCESS_KEY']
            self.region = match.get('REGION', None)
        if self.aws_access_key_id is None or self.aws_secret_access_key is None:
            raise IOError('No AWS credentials found')

        self.getS3Session()

    def getS3Session(self):

        if self.__s3_session is None:
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            self.__s3_session = session

        return self.__s3_session

    def getS3Client(self):
        if self.__s3_session is None:
            session = self.getS3Session()
        else:
            session = self.__s3_session
        return session.client(u's3')

    def get_files_from_s3(self, alt_domain=None):
        try:
            documents = []
            doma = self.domain
            if alt_domain:
                doma = alt_domain
            for doc in self.__get_all_s3_objects(Bucket=self.bucket, Prefix=doma):
                documents.append(S3File(doc))

            return documents
        except Exception as e:
            raise e

    def __get_all_s3_objects(self, **base_kwargs):
        continuation_token = None
        while True:
            list_kwargs = dict(MaxKeys=1000, **base_kwargs)
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token

            client = self.getS3Client()
            response = client.list_objects_v2(**list_kwargs)
            yield from response.get('Contents', [])
            if not response.get('IsTruncated'):  # At the end of the list?
                break
            continuation_token = response.get('NextContinuationToken')

    def get_txt_file(self, key, retry=True):
        try:
            bytes_buffer = io.BytesIO()
            client = self.getS3Client()
            client.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=bytes_buffer)
            data_decoded = bytes_buffer.getvalue()
            return base64.b64decode(data_decoded).decode('utf-8')
        except Exception as e:
            if retry:
                return self.get_txt_file(key, False)
            raise S3FileNotFoundException(key)

    def get_byte_file(self, key, retry=True):
        try:
            bytes_buffer = io.BytesIO()
            client = self.getS3Client()
            client.download_fileobj(Bucket=self.bucket, Key=key, Fileobj=bytes_buffer)
            data_decoded = bytes_buffer.getvalue()
            return data_decoded
        except Exception as e:
            if retry:
                return self.get_byte_file(key, False)
            raise S3FileNotFoundException(key)

    def check_file(self, key):
        s3 = boto3.resource('s3',
                            aws_access_key_id=self.aws_access_key_id,
                            aws_secret_access_key=self.aws_secret_access_key)
        try:
            s3.Object(self.bucket, key).load()
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False  # The object does not exist.
            else:
                raise  # Something else has gone wrong.
        else:
            return True  # The object does exist.

    def upload_file(self, key, content):
        client = self.getS3Client()
        response = client.put_object(
            Bucket=self.bucket,
            Key=str(key),
            Body=content
        )
        return response

    def upload_file_by_path(self, file_name, key):
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param key: S3 object name
        :return: True if file was uploaded, else False
        """

        # If S3 object_name was not specified, use file_name
        client = self.getS3Client()
        try:
            response = client.upload_file(file_name, self.bucket, key)
        except Exception:
            return False
        return True

    @staticmethod
    def s3_check_by_extension(s3_elements, extension):
        extension = extension.upper()
        for obj in s3_elements:
            if obj.Size > 0 and obj.Key.upper().endswith(f'.{extension}'):
                return True
        return False
