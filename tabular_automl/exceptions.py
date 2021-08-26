"""Custom exceptions for tabular automl"""


class Error(Exception):
    """Base class for other exceptions"""


class UnsupportedFileFormatError(Error):
    """Raised when the file format is unsupported"""


class UnsupportedTaskTypeError(Error):
    """Raised when the task type is unsupported"""
