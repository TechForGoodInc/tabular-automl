import io
import unittest

import pandas as pd

from tabular_automl import TabularData, settings
from tabular_automl.exceptions import UnsupportedFileFormatError


class TabularDataTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = settings.DATA_DIR / "titanic"
        cls.data_path = cls.data_dir / "train.csv"
        cls.index_col = "PassengerId"

    def setUp(self):
        self.data_buffer = io.open(self.data_path, "rb")

    def tearDown(self):
        self.data_buffer.close()

    def test_get_filepath_or_buffer_extension(self):
        pathlib_dataset = TabularData(self.data_path, index_col=self.index_col)
        self.assertEqual(pathlib_dataset._get_filepath_or_buffer_extension(), ".csv")

        buffer_dataset = TabularData(self.data_buffer, index_col=self.index_col)
        self.assertEqual(buffer_dataset._get_filepath_or_buffer_extension(), ".csv")

    def test_get_data(self):
        # non-existent path
        with self.assertRaises(FileNotFoundError):
            TabularData(self.data_dir / "does_not_exist.csv")

        # unsupported format
        with self.assertRaises(UnsupportedFileFormatError):
            TabularData(self.data_dir / "titanic.zip")

        # non-existent `index_col`
        with self.assertRaises(ValueError):
            TabularData(self.data_path, index_col="unknown")

        dataset = TabularData(self.data_path, index_col=self.index_col)
        self.assertTrue(isinstance(dataset.data, pd.DataFrame))

        dataset = TabularData(self.data_buffer, index_col=self.index_col)
        self.assertTrue(isinstance(dataset.data, pd.DataFrame))
