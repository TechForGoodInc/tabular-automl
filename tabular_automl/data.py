from pathlib import Path

from . import settings
from .exceptions import UnsupportedFileFormatError


class TabularData:
    """Wrapper around pandas functions for manipulating tabular data"""

    def __init__(self, filepath_or_buffer, index_col=None):
        self.filepath_or_buffer = filepath_or_buffer
        self.data = self.get_data()
        if index_col is not None:
            self.set_index(index_col)

    def _get_filepath_or_buffer_extension(self):
        try:
            return Path(self.filepath_or_buffer).suffix
        except TypeError:
            return Path(self.filepath_or_buffer.name).suffix

    def get_data(self):
        try:
            ext = self._get_filepath_or_buffer_extension()
            read_func = settings.FILE_READERS.get(ext)
            return read_func(self.filepath_or_buffer)
        except TypeError:
            raise UnsupportedFileFormatError

    def set_index(self, index_col):
        try:
            # check if `index_col` is in the data
            index = list(set(self.data.columns) & {index_col})
            data = self.data.set_index(index)
            self.data = data
            self.index_col = index_col
        except KeyError:
            raise ValueError(f"{index_col} not found in data")
