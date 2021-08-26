# tabular-automl
A low code, low-cost AutoML solution for tabular data.

[![Tests](https://github.com/harisonmg/tabular-automl/actions/workflows/ci.yml/badge.svg)](https://github.com/harisonmg/tabular-automl/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/harisonmg/tabular-automl/badge.svg?branch=main)](https://coveralls.io/github/harisonmg/tabular-automl?branch=main)

## Installation and usage

`tabular-automl` is available as a Python package, as well as a web application in a docker container.

### Using the web application

1. Pull the container from one of the supported registries
    ```
    # docker hub
    $ docker pull harisonmg/tabular-automl:latest

    # GitHub packages
    $ docker pull ghcr.io/harisonmg/tabular-automl:latest
    ```
1. Run the container \
    `$ docker run -dp 8000:8000 tabular-automl:latest`
1. Visit [localhost:8000](http://localhost:8000) in your browser
1. Select a task type and upload your data
1. Wait for a few minutes and you'll get a trained model as well as sample predictions


### Using the Python package

This assumes you have basic knowledge of Python.

1. Install the package from PyPI
    ```
    # using pip
    $ pip install tabular-automl

    # using pipenv
    $ pipenv install tabular-automl --skip-lock
    ```

2. View example usage in the following Kaggle notebook: \
    https://www.kaggle.com/harisonmwangi/tabular-automl/


## License

The project is licensed under the MIT license.
