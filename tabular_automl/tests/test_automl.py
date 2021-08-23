import unittest

from tabular_automl import settings, TabularAutoML, TabularData
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
        train_dataset = TabularData(
            self.train_data_path, index_col=self.index_col
        )
        self.train_data = train_dataset.data

    @staticmethod
    def get_sample_frac(data, sample):
        return sample.shape[0] / data.shape[0]

    def test_get_module(self):
        # None task type
        with self.assertRaises(ValueError):
            TabularAutoML(self.train_data, target_col=self.target_col)

        # unsupported task type
        with self.assertRaises(UnsupportedTaskTypeError):
            TabularAutoML(
                self.train_data,
                target_col=self.target_col,
                task_type="unknown"
            )

        automl = TabularAutoML(
            self.train_data,
            target_col=self.target_col,
            task_type=self.task_type
        )
        self.assertEqual(
            automl.pycaret_module.__name__, "pycaret.classification"
        )

    def test_get_sample(self):
        automl = TabularAutoML(
            self.train_data,
            target_col=self.target_col,
            task_type=self.task_type,
        )

        sample_frac = self.get_sample_frac(
            self.train_data, sample=automl.get_sample()
        )
        self.assertGreaterEqual(sample_frac, 0.5)

    def test_get_best_model(self):
        automl = TabularAutoML(
            self.train_data,
            target_col=self.target_col,
            task_type=self.task_type,
        )
        config = {"setup": dict(silent=True)}
        automl.get_best_model(config)
