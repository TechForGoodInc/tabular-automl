import streamlit as st

from tabular_automl import settings as automl_settings
from tabular_automl import TabularAutoML, TabularData

import settings as app_settings

# app display
st.title(app_settings.APP_NAME)

task_type = st.radio(
    "Choose a task type", options=automl_settings.SUPPORTED_TASK_TYPES
)

def get_dataset(subset="training"):
    # data upload
    data_buffer = st.file_uploader(
        f"Upload {subset} data", type=automl_settings.FILE_READERS.keys()
    )
    if data_buffer is not None:
        return TabularData(data_buffer)

def get_data():
    train_data = get_dataset()
    test_data = get_dataset(subset="testing")

    if train_data is not None:
        with st.expander("Does the data have an index column?"):
            index_col = st.selectbox(
                "Select the index column", train_data.data.columns
            )
            train_data.set_index(index_col)
            if test_data is not None:
                test_data.set_index(index_col)

        if test_data is None:
            return train_data.data, None
        else:
            return train_data.data, test_data.data    

@st.cache
def create_experiment(train_data, test_data=None):
    automl = TabularAutoML(
        train_data,
        test_data=test_data,
        target_col=target_col,
        task_type=task_type
    )
    return automl

def get_model_outputs_dir():
    dir = automl_settings.MODEL_OUTPUTS_DIR
    if not dir.exists():
        dir.mkdir(parents=True)
    return dir

data = get_data()
if data is not None:
    train_data, test_data = data
    target_col = st.selectbox(
        "Select the target column", train_data.columns
    )
    with st.spinner(text="Setting up"):
        automl = create_experiment(train_data, test_data=test_data)
        model = None
        config = {
            # "sampling": dict(sample_frac=round(1/3, 2)),
            "setup": dict(silent=True),
        }

    if st.button("Start training"):        
        cached_get_best_model = st.cache(automl.get_best_model)
        cached_tune_model = st.cache(automl.tune_model)
        cached_finalize_model = st.cache(automl.finalize_model)
        with st.spinner(text="Training in progress ..."):
            best_model = cached_get_best_model(config)
            tuned_model = cached_tune_model(estimator=best_model)
            final_model = cached_finalize_model(estimator=tuned_model)
            model = final_model
            st.success("Training successfully completed!")

    if model is not None:
        model_outputs_dir = get_model_outputs_dir()
        predictions = automl.predict_model(estimator=model, data=test_data)
        st.write("Sample predictions", predictions)
        predictions_file = model_outputs_dir / "predictions.csv"
        predictions.to_csv(predictions_file)
