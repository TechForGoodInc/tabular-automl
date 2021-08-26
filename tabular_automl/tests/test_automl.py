import unittest

from tabular_automl import TabularAutoML, TabularData, settings
from tabular_automl.exceptions import UnsupportedTaskTypeError


class TabularAutoMLTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = settings.DATA_DIR / "titanic"
        cls.train_data_path = cls.data_dir / "train.csv"
        cls.index_col = "PassengerId"
        cls.target_col = "Survived"
        cls.task_type = "classification"

    def setUp(self):
        train_dataset = TabularData(self.train_data_path, index_col=self.index_col)
        self.train_data = train_dataset.data

    @staticmethod
    def get_sample_frac(data, sample):
        return round(sample.shape[0] / data.shape[0], 2)

    def test_get_module(self):
        # None task type
        with self.assertRaises(ValueError):
            TabularAutoML(self.train_data, target_col=self.target_col)

        # unsupported task type
        with self.assertRaises(UnsupportedTaskTypeError):
            TabularAutoML(
                self.train_data, target_col=self.target_col, task_type="unknown"
            )

        pipeline = TabularAutoML(
            self.train_data, target_col=self.target_col, task_type=self.task_type
        )
        self.assertEqual(pipeline.pycaret_module.__name__, "pycaret.classification")

    def test_get_sample(self):
        pipeline = TabularAutoML(
            self.train_data,
            target_col=self.target_col,
            task_type=self.task_type,
        )

        sample_frac = self.get_sample_frac(
            self.train_data, sample=pipeline.get_sample()
        )
        self.assertAlmostEqual(sample_frac, 0.5)

        sample_frac = self.get_sample_frac(
            self.train_data, sample=pipeline.get_sample(sample_frac=0.1)
        )
        self.assertAlmostEqual(sample_frac, 0.1)

    def test_get_best_model(self):
        pipeline = TabularAutoML(
            self.train_data,
            target_col=self.target_col,
            task_type=self.task_type,
        )
        config = {"setup": dict(silent=True)}
        pipeline.get_best_model(config)
