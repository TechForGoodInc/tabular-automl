from pathlib import Path

import unittest

import pandas as pd

from automl.settings import DATA_DIR
from automl.tabular import TabularAutoML


class TabularAutoMLTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = DATA_DIR / "titanic"
        cls.train_data_dir = cls.data_dir / "train.csv"
        cls.index = "PassengerId"
        cls.target = "Survived"
        cls.task = "classification"

        print(f"index_col {cls.index}")

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
                self.train_data_dir,
                target_col=self.target
            )

        with self.assertRaises(ValueError):
            TabularAutoML(
                self.train_data_dir,
                target_col=self.target,
                task_type="unknown"
            )

    def test_get_data(self):
        with self.assertRaises(ValueError):
            TabularAutoML(
                self.train_data_dir,
                index_col="unknown",
                target_col=self.target,
                task_type=self.task
            )
        
        with self.assertRaises(ValueError):
            TabularAutoML(
                self.train_data_dir,
                index_col=self.index,
                target_col="unknown",
                task_type=self.task
            )

        automl = TabularAutoML(
            self.train_data_dir,
            index_col=self.index,
            target_col=self.target,
            task_type=self.task
        )
        self.assertTrue(isinstance(automl.data, pd.DataFrame))

    def test_get_sample(self):
        automl = TabularAutoML(
            self.train_data_dir,
            index_col=self.index,
            target_col=self.target,
            task_type=self.task
        )
        sample= automl._get_sample()
        self.assertEqual(sample.shape, (89,12))

    def test_setup(self):
        pass

    def test_compare_models(self):
        pass

    def test_get_best_model(self):
        automl = TabularAutoML(
            self.train_data_dir,
            index_col=self.index,
            target_col=self.target,
            task_type=self.task
        )
        config = {"setup": dict(silent=True)}
        automl.get_best_model(config)


if __name__ == "__main__":
    unittest.main()
