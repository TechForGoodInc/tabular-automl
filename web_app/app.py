import streamlit as st
from settings import APP_NAME

from tabular_automl import TabularAutoML, TabularData
from tabular_automl.settings import FILE_READERS, SUPPORTED_TASK_TYPES

# app display
st.title(APP_NAME)

task_type = st.radio("Choose a task type", options=SUPPORTED_TASK_TYPES)


def get_dataset(subset="training"):
    # data upload
    data_buffer = st.file_uploader(f"Upload {subset} data", type=FILE_READERS.keys())
    if data_buffer is not None:
        return TabularData(data_buffer)
    return None


def get_data():
    train_dataset = get_dataset()
    test_dataset = get_dataset(subset="testing")

    if train_dataset is not None:
        with st.expander("Does the data have an index column?"):
            index_col = st.selectbox(
                "Select the index column", train_dataset.data.columns
            )
            train_dataset.set_index(index_col)
            if test_dataset is not None:
                test_dataset.set_index(index_col)

        if test_dataset is None:
            return train_dataset.data, None
        return train_dataset.data, test_dataset.data
    return None


@st.cache
def create_pipeline():
    return TabularAutoML(
        train_data, test_data=test_data, target_col=target_col, task_type=task_type
    )


data = get_data()
if data is not None:
    train_data, test_data = data
    target_col = st.selectbox("Select the target column", train_data.columns)
    with st.spinner(text="Setting up"):
        pipeline = create_pipeline()
        model = None
        config = {
            # "sampling": dict(sample_frac=round(1/3, 2)),
            "setup": dict(silent=True),
        }

    if st.button("Start training"):
        cached_get_best_model = st.cache(pipeline.get_best_model)
        cached_tune_model = st.cache(pipeline.tune_model)
        cached_finalize_model = st.cache(pipeline.finalize_model)
        with st.spinner(text="Training in progress ..."):
            best_model = cached_get_best_model(config)
            tuned_model = cached_tune_model(estimator=best_model)
            final_model = cached_finalize_model(estimator=tuned_model)
            model = final_model
            st.success("Training successfully completed!")

    if model is not None:
        predictions = pipeline.predict_model(estimator=model, data=test_data)
        st.write("Sample predictions", predictions)
