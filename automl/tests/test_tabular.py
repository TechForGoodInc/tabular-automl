from pathlib import Path

import unittest

import pandas as pd

from automl.settings import DATA_DIR
from automl.tabular import TabularAutoML


class TabularAutoMLTestCase(unittest.TestCase):
    def setUp(self):
        self.data_dir = DATA_DIR / "titanic"
        self.train_data = self.data_dir / "train.csv"
        self.index_col = "PassengerId"
        self.target_col = "Survived"
        self.task_type = "classification"

        print(f"DATA_DIR contents: {list(DATA_DIR.rglob('*'))}")

    def test_basic(self):
        with self.assertRaises(TypeError):
            TabularAutoML()

    def test_validate_file_path(self):
        # not a Path
        with self.assertRaises(ValueError):
            TabularAutoML("not_a_path")
        
        # non-existent path
        with self.assertRaises(ValueError):
            TabularAutoML(self.data_dir / "does_not_exist.csv")

        # unsupported format
        with self.assertRaises(ValueError):
            TabularAutoML(self.data_dir / "titanic.zip")

    def test_get_module(self):
        with self.assertRaises(ValueError):
            TabularAutoML(
                self.train_data,
                target_col=self.target_col
            )

        with self.assertRaises(ValueError):
            TabularAutoML(
                self.train_data,
                target_col=self.target_col,
                task_type="unknown"
            )

    def test_get_data(self):
        with self.assertRaises(ValueError):
            TabularAutoML(
                self.train_data,
                index_col="unknown",
                target_col=self.target_col,
                task_type=self.task_type
            )
        
        with self.assertRaises(ValueError):
            TabularAutoML(
                self.train_data,
                index_col=self.index_col,
                target_col="unknown",
                task_type=self.task_type
            )

        automl = TabularAutoML(
            self.train_data,
            index_col=self.index_col,
            target_col=self.target_col,
            task_type=self.task_type
        )
        self.assertTrue(isinstance(automl.data, pd.DataFrame))

    def test_get_sample(self):
        automl = TabularAutoML(
            self.train_data,
            index_col=self.index_col,
            target_col=self.target_col,
            task_type=self.task_type
        )
        sample= automl._get_sample()
        self.assertEqual(sample.shape, (89,12))

    def test_setup(self):
        pass

    def test_compare_models(self):
        pass

    def test_get_best_model(self):
        automl = TabularAutoML(
            self.train_data,
            index_col=self.index_col,
            target_col=self.target_col,
            task_type=self.task_type
        )
        config = {"setup": dict(silent=True)}
        automl.get_best_model(config)


if __name__ == "__main__":
    unittest.main()
