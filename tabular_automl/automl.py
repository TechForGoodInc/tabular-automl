import copy

from IPython.display import display

from .exceptions import UnsupportedTaskTypeError
from .settings import LARGE_DATASET_ROWS, SUPPORTED_TASK_TYPES


class TabularAutoML:
    """Wrapper around PyCaret for machine learning tasks
    that use tabular data
    """

    def __init__(self, train_data, test_data=None, target_col=None, task_type=None):
        self.train_data = train_data
        self.test_data = test_data
        self.target_col = target_col
        self.task_type = task_type

        # derived attributes
        self.pycaret_module = self.get_module()

    def get_module(self):
        try:
            import importlib

            module = importlib.import_module(f"pycaret.{str(self.task_type)}")
            return module
        except ModuleNotFoundError:
            if self.task_type is None:
                raise ValueError("Task type must be set!")

            if self.task_type not in SUPPORTED_TASK_TYPES:
                raise UnsupportedTaskTypeError

    def setup(self, **kwargs):
        return self.pycaret_module.setup(**kwargs)

    def compare_models(self, **kwargs):
        return self.pycaret_module.compare_models(**kwargs)

    def create_model(self, **kwargs):
        return self.pycaret_module.create_model(**kwargs)

    def tune_model(self, **kwargs):
        return self.pycaret_module.tune_model(**kwargs)

    def plot_model(self, **kwargs):
        return self.pycaret_module.plot_model(**kwargs)

    def interpret_model(self, **kwargs):
        return self.pycaret_module.interpret_model(**kwargs)

    def predict_model(self, **kwargs):
        return self.pycaret_module.predict_model(**kwargs)

    def finalize_model(self, **kwargs):
        return self.pycaret_module.finalize_model(**kwargs)

    def save_model(self, **kwargs):
        return self.pycaret_module.save_model(**kwargs)

    def load_model(self, **kwargs):
        return self.pycaret_module.load_model(**kwargs)

    def get_sample(self, subset="train", sample_frac="auto", random_state=42):
        if subset == "train":
            data = self.train_data
        elif subset == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid subset: {subset}")

        data_rows = data.shape[0]
        if sample_frac == "auto":
            if data_rows > LARGE_DATASET_ROWS:
                sample_frac = LARGE_DATASET_ROWS / data_rows
                sample_frac = round(sample_frac, 2)
            else:
                sample_frac = 0.5

        sample_data = data.sample(frac=sample_frac, random_state=random_state)
        return sample_data

    def get_best_model(self, config=None):
        if config is None:
            config = {}
        else:
            config = copy.deepcopy(config)
        sampling__config = config.get("sampling", {})
        setup__config = config.get("setup", {})
        compare_models__config = config.get("compare_models", {})

        # get a sample if there is a sampling config
        # or the dataset is large
        large_dataset = self.train_data.shape[0] / LARGE_DATASET_ROWS
        if bool(sampling__config) or large_dataset:
            data = self.get_sample(**sampling__config)
        else:
            data = self.train_data

        # initialize the experiment
        # override the data given in the config
        setup__config["data"] = data
        if setup__config.get("target") is None:
            setup__config["target"] = self.target_col
        setup = self.setup(**setup__config)

        for item in setup:
            display(item)

        # run the experiment
        best_model = self.compare_models(**compare_models__config)
        return best_model
