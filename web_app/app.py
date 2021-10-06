import streamlit as st
from settings import APP_NAME

from tabular_automl import TabularAutoML, TabularData
from tabular_automl.settings import FILE_READERS, SUPPORTED_TASK_TYPES

# app display
st.title(APP_NAME)

task_type = st.radio("Choose a task type", options=SUPPORTED_TASK_TYPES)


def _get_dataset(subset="training"):
    # data upload
    data_buffer = st.file_uploader(f"Upload {subset} data", type=FILE_READERS.keys())
    if data_buffer is not None:
        return TabularData(data_buffer)
    return None


def get_data():
    train_dataset = _get_dataset()
    test_dataset = _get_dataset(subset="testing")

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


def train_model(pipeline):
    with st.spinner(text="Training in progress ..."):
        config = {
            # "sampling": dict(sample_frac=round(1/3, 2)),
            "setup": dict(silent=True),
        }
        best_model = st.cache(pipeline.get_best_model)(config)
        tuned_model = st.cache(pipeline.tune_model)(estimator=best_model)
        final_model = st.cache(pipeline.finalize_model)(estimator=tuned_model)
        model = final_model
        st.success("Training successfully completed!")
    return model


def main():
    data = get_data()
    if data is not None:
        train_data, test_data = data
        target_col = st.selectbox("Select the target column", train_data.columns)
        with st.spinner(text="Setting up"):
            pipeline = st.cache(TabularAutoML)(
                train_data,
                test_data=test_data,
                target_col=target_col,
                task_type=task_type,
            )

        if st.button("Start training"):
            # model
            model = train_model(pipeline)

            # predictions
            predictions = pipeline.predict_model(estimator=model, data=test_data)
            st.subheader("Sample predictions")
            st.write(predictions.sample(10))

            # downloads
            st.subheader("Downloads")
            st.download_button(
                "Download predictions",
                data=predictions.to_csv(),
                file_name="predictions.csv",
                mime="text/csv",
            )


main()
