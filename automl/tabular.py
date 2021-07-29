from pathlib import Path

import pandas as pd

# constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class TabularAutoML(object):
    """
    Wrapper around PyCaret for machine learning tasks that
    use tabular data
    """

    TASK_TYPES = ("regression", "classification")
    SUPPORTED_FILE_FORMATS = (".csv",)

    def __init__(self, data_file_path, index_col=None, target_col=None, task_type=None):
        # TODO: handle multiple file paths
        self.data_file_path = self._validate_file_path(data_file_path)
        self.data, self.target_col = self.get_data(
            self.data_file_path, index_col=index_col, target_col=target_col
        )
        self._module = self._get_module(task_type)

    def _validate_file_path(self, file_path):
        if not isinstance(file_path, Path):
            raise ValueError("`data_file_path` must be a `pathlib.Path` object")
        elif not file_path.exists():
            raise ValueError("File not found!")
        elif file_path.suffix not in self.SUPPORTED_FILE_FORMATS:
            raise ValueError("Unsupported file format!")
        return file_path

    def _get_module(self, task_type):
        try:
            import importlib

            module = importlib.import_module(f"pycaret.{task_type}")
            return module
        except ImportError:
            if task_type not in self.TASK_TYPES:
                raise ValueError(f"Unsupported task type: {task_type}")

    def get_data(self, file_path, index_col=None, target_col=None):
        # TODO: handle other file extensions
        data = pd.read_csv(file_path)

        # check if the index column is in the data
        try:
            index = list(set(data.columns) & set([self.index_col]))
            data.set_index(index)
        except KeyError:
            if index_col is not None:
                raise ValueError(f"{index_col} not found in data")

        # check if the target column is in the data
        if target_col not in data.columns:
            raise ValueError(f"{target_col} not found in data")

        return data, target_col

    def _get_sample(self, sample_frac=0.1, random_state=42):
        sample_data = self.data.sample(frac=sample_frac, random_state=random_state)
        print(f"Sample shape: {sample_data.shape}")
        return sample_data

    def setup(self, data, target, **kwargs):
        return self._module.setup(data, target=target, **kwargs)

    def compare_models(self, **kwargs):
        return self._module.compare_models(**kwargs)

    def get_best_model(self, config=None):
        if config is None:
            config = {}
        sampling__config = config.get("sampling", {})
        setup__config = config.get("setup", {})
        compare_models__config = config.get("compare_models", {})

        # get a sample if the data is large
        if self.data.shape[0] > 10000:
            data = self._get_sample(**sampling__config)
        else:
            data = self.data

        # initialize the experiment
        setup__config.update(dict(data=data, target=self.target_col))
        self.setup(**setup__config)

        # run the experiment
        best_model = self.compare_models(**compare_models__config)
        return best_model


if __name__ == "__main__":
    # list all file paths in DATA_DIR and its subdirectories
    for file_path in DATA_DIR.rglob("*"):
        print(file_path)

    data_file_path = input("Enter a path to the dataset: ")
    index_col = input("Enter the index column: ")
    target_col = input("Enter the index column: ")
    task_type = input("Choose a task type: ")

    automl = TabularAutoML(
        data_file_path, index_col=index_col, target_col=target_col, task_type=task_type
    )
    config = {
        "sampling": dict(sample_frac=0.01),
        "setup": dict(silent=True),
        "compare_models": {},
    }
    automl.get_best_model(config)
