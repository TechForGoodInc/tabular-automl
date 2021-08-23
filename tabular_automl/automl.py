from . import settings
from .exceptions import UnsupportedTaskTypeError


class TabularAutoML:
    """
    Wrapper around PyCaret for machine learning tasks that
    use tabular data
    """
    def __init__(self, train_data, target_col=None, task_type=None):
        # TODO: add option for test data
        self.train_data = train_data
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

            if self.task_type not in settings.SUPPORTED_TASK_TYPES:
                raise UnsupportedTaskTypeError

    def setup(self, **kwargs):
        return self.pycaret_module.setup(**kwargs)

    def compare_models(self, **kwargs):
        return self.pycaret_module.compare_models(**kwargs)

    def get_sample(self, sample_frac="auto", random_state=42):
        train_data_rows = self.train_data.shape[0]

        if sample_frac == "auto":
            if train_data_rows > settings.LARGE_DATASET_ROWS:
                sample_frac = settings.LARGE_DATASET_ROWS / train_data_rows
                sample_frac = round(sample_frac, 2)
            else:
                sample_frac = 0.5

        sample_data = self.train_data.sample(
            frac=sample_frac, random_state=random_state
        )
        return sample_data

    def get_best_model(self, config=None):
        if config is None:
            config = {}
        sampling__config = config.get("sampling", {})
        setup__config = config.get("setup", {})
        compare_models__config = config.get("compare_models", {})

        # get a sample if the data is large
        if self.train_data.shape[0] > settings.LARGE_DATASET_ROWS:
            data = self.get_sample(**sampling__config)
        else:
            data = self.train_data

        # initialize the experiment
        # override the data given in the config
        setup__config["data"] = data
        if setup__config.get("target") is None:
            setup__config["target"] = self.target_col
        setup = self.setup(**setup__config)
        # print(setup)

        # run the experiment
        best_model = self.compare_models(**compare_models__config)
        return best_model