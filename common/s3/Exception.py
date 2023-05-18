class S3FileException(Exception):
    def __init__(self, key, bucket):
        super(S3FileException, self).__init__(
            f"The file '{key}' is not reachable at S3:{bucket}")


class S3FileTooBigException(Exception):
    def __init__(self, key, size):
        super(S3FileTooBigException, self).__init__(
            f"The file '{key}' is too big to be processed. Size is {size} bytes")


class NoDataException(Exception):
    def __init__(self, key, fields=None):
        if fields is not None:
            super(NoDataException, self).__init__(
                f"Couldn't find one or more critical fields in '{key}': {str(fields)}")
        else:
            super(NoDataException, self).__init__(
                f"Couldn't find one or more critical fields in '{key}'")


class NoCertificationFoundException(Exception):
    def __init__(self, key):
        super(NoCertificationFoundException, self).__init__(
            f"Couldn't find appraisal certification in '{key}'")


class SmallFileException(Exception):
    def __init__(self, key, length):
        super(SmallFileException, self).__init__(
            f"File '{key}' is too small with length {length}")


class NotReadableFileException(Exception):
    def __init__(self, key, ratio):
        super(NotReadableFileException, self).__init__(
            f"File '{key}' is not readable in spanish with {ratio}/sp")


class FileFormatException(Exception):
    def __init__(self, key):
        super(FileFormatException, self).__init__(
            f"File '{key}' format is not allowed")


class InvalidFileException(Exception):
    def __init__(self, key):
        super(InvalidFileException, self).__init__(
            f"File '{key}' is not a readable document")


class S3FileNotFoundException(Exception):
    def __init__(self, key):
        # Call the base class constructor with the parameters it needs
        super(S3FileNotFoundException, self).__init__(f"The key '{key}' was not found")


class NoCertificateException(Exception):
    def __init__(self, key):
        # Call the base class constructor with the parameters it needs
        super(NoCertificateException, self).__init__(f"{key}' doesn't contain certificate")
