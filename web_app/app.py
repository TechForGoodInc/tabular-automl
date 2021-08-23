import streamlit as st

from tabular_automl import settings as automl_settings
from tabular_automl import TabularAutoML, TabularData

import settings as app_settings

# app display
st.title(app_settings.APP_NAME)

# general configuration
task_type = st.radio(
    "Choose a task type", options=automl_settings.SUPPORTED_TASK_TYPES
)

automl_config = {
    "sampling": dict(sample_frac=round(1/3, 2)),
    "setup": dict(silent=True),
}

def get_data():
    # training data upload
    training_data_buffer = st.file_uploader(
        "Upload training data", type=automl_settings.FILE_READERS.keys()
    )
    if training_data_buffer is not None:
        # data operations
        training_data = TabularData(training_data_buffer)

        with st.expander("Does the data have an index column?"):
            index_col = st.selectbox(
                "Select the index column", training_data.data.columns
            )
            training_data.set_index(index_col)

        return training_data.data

@st.cache
def train_model(training_data, target_col):
    automl = TabularAutoML(
        training_data, target_col=target_col, task_type=task_type
    )
    best_model = automl.get_best_model(automl_config)
    return best_model

training_data = get_data()
if training_data is not None:
    target_col = st.selectbox(
        "Select the target column", training_data.columns
    )

    if st.button("Start training"):
        best_model = train_model(training_data, target_col)
        best_model
