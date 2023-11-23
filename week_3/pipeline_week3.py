from kfp.v2.dsl import component, pipeline
from kfp.v2.dsl import Input, Output, Metrics, Artifact, Model, HTML, Dataset
from typing import NamedTuple

VERTEX_PIPELINES_ROOT = "gs://[PUT-YOUR-GCS-BUCKET-HERE]/"
PIPELINE_NAME = "mlops-retraining-w-1-2"


@component(base_image='python:3.8', packages_to_install=["scikit-learn==1.2.2", "pandas==1.5.3"])
def sklearn_to_gcs(dataset_artifact: Output[Dataset]):
    from sklearn.datasets import load_wine

    df = load_wine(as_frame=True)
    df = df["data"].join(df["target"])

    df.to_csv(f"{dataset_artifact.path}.csv", index=False, sep=",")


@component(base_image='python:3.8', packages_to_install=["scikit-learn==1.2.2", "pandas==1.5.3"])
def preprocessing(dataset_artifact: Input[Dataset], dataset_artifact_train: Output[Dataset],
                  dataset_artifact_test: Output[Dataset], dataset_artifact_inf: Output[Dataset],
                  test_size: float = 0.33, random_state: int = 42):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(f"{dataset_artifact.path}.csv", sep=",")

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    df_test, df_inf = train_test_split(
        df, test_size=0.1, random_state=random_state
    )

    df_train.to_csv(f"{dataset_artifact_train.path}.csv", index=False, sep=",")
    df_test.to_csv(f"{dataset_artifact_test.path}.csv", index=False, sep=",")
    df_inf.to_csv(f"{dataset_artifact_inf.path}.csv", index=False, sep=",")


@component(base_image='python:3.8', packages_to_install=["tensorflow_data_validation==1.12.0", "pandas==1.5.3"])
def tfdv_generate_schema(dataset_artifact: Input[Dataset], label_column: str, statistics_html: Output[HTML],
                         features_html: Output[HTML], cardinality_html: Output[HTML],
                         schema_artifact: Output[Artifact], split_column: str = "None"):
    import tensorflow_data_validation as tfdv
    from tensorflow_data_validation.utils.display_util import get_statistics_html
    from tensorflow_data_validation.utils.display_util import get_schema_dataframe

    statistics = tfdv.generate_statistics_from_csv(f"{dataset_artifact.path}.csv")
    html = get_statistics_html(statistics)
    with open(statistics_html.path, "w") as f:
        f.write(html)

    schema = tfdv.infer_schema(statistics)
    schema.default_environment.append("TRAINING")
    schema.default_environment.append("SERVING")

    if split_column != "None":
        tfdv.get_feature(schema, split_column).not_in_environment.append("SERVING")
        tfdv.get_feature(schema, split_column).not_in_environment.append("TRAINING")

    tfdv.get_feature(schema, label_column).not_in_environment.append("SERVING")

    tfdv.write_schema_text(schema, schema_artifact.path)

    features, cardinality = get_schema_dataframe(schema)
    for df, output in zip([features, cardinality], [features_html, cardinality_html]):
        html = df.to_html()
        with open(output.path, "w") as f:
            f.write(html)


@component(base_image='python:3.8',
           packages_to_install=["tensorflow_data_validation==1.12.0", "pandas==1.5.3", "IPython==8.11.0"])
def tfdv_validate_with_schema(dataset_artifact: Input[Dataset], statistics_html: Output[HTML],
                              frozen_schema_path: Input[Artifact], anomalies_html: Output[HTML], ):
    import tensorflow_data_validation as tfdv
    from tensorflow_data_validation.utils.display_util import get_statistics_html
    from tensorflow_data_validation.utils.display_util import get_anomalies_dataframe

    statistics = tfdv.generate_statistics_from_csv(f"{dataset_artifact.path}.csv")
    html = get_statistics_html(statistics)
    with open(statistics_html.path, "w") as f:
        f.write(html)

    schema = tfdv.load_schema_text(frozen_schema_path.path)
    anomalies = tfdv.validate_statistics(statistics, schema)
    df = get_anomalies_dataframe(anomalies)
    html = df.to_html()
    with open(anomalies_html.path, "w") as f:
        f.write(html)


@component(base_image='python:3.8', packages_to_install=["scikit-learn==1.2.2", "pandas==1.5.3", "xgboost==1.7.4"])
def train(dataset_artifact: Input[Dataset], model: Output[Model], label_column: str):
    import xgboost as xgb
    import pandas as pd
    import pickle

    X = pd.read_csv(f"{dataset_artifact.path}.csv", sep=",")
    y = X[label_column]
    X.drop(columns=[label_column], inplace=True)

    model_xgb = xgb.XGBClassifier()
    model_xgb.fit(X, y)

    file_name = model.path + ".pkl"
    with open(file_name, "wb") as file:
        pickle.dump(model_xgb, file)


@component(base_image='python:3.8', packages_to_install=["scikit-learn==1.2.2", "pandas==1.5.3", "xgboost==1.7.4"])
def evaluate(model: Input[Model], dataset_artifact: Input[Dataset], metrics: Output[Metrics], label_column: str):
    import xgboost as xgb
    import pickle
    from sklearn.metrics import accuracy_score
    import pandas as pd

    model_xgb = xgb.XGBClassifier()
    file_name = model.path + ".pkl"
    with open(file_name, "rb") as file:
        model_xgb = pickle.load(file)

    X = pd.read_csv(f"{dataset_artifact.path}.csv", sep=",")
    y_true = X[label_column]
    X.drop(columns=[label_column], inplace=True)

    y_pred = model_xgb.predict(X)

    accuracy = accuracy_score(y_true, y_pred)

    metrics.log_metric("accuracy", accuracy)


@component(base_image='python:3.8', packages_to_install=["google-cloud-aiplatform==1.22.1"])
def deploy_model(display_name: str, model: Input[Model], serving_container: str,
                 project_id: str, location: str) -> NamedTuple("Outputs", [("model_resource_name", str)]):
    from google.cloud import aiplatform as vertex_ai
    import logging

    logging.getLogger().setLevel(logging.INFO)

    model = vertex_ai.Model.upload(
        display_name=display_name,
        artifact_uri=model.uri.replace("model", ""),
        serving_container_image_uri=serving_container,
        location=location,
        project=project_id,
    )
    logging.info("Model was deployed")

    return (model.resource_name,)


@component(base_image='python:3.8', packages_to_install=['pandas==1.5.3', 'fsspec==2023.3.0', 'gcsfs==2023.3.0'])
def preprocess_batch(batch_data_gcs: Input[Dataset], preprocessed_batch_artifact: Output[Dataset]):
    import pandas as pd
    import csv

    df = pd.read_csv(f"{batch_data_gcs.path}.csv", sep=",")

    # wrapping all strings into double quotes
    for col in df.columns:
        if df[col].dtype in ['O', 'string[python]']:
            df[col] = df[col].apply(lambda x: '"' + x + '"')

    col_names = df.columns
    fixed_col_names = ['"' + col + '"' for col in col_names]
    df.columns = fixed_col_names

    df.to_csv(f"{preprocessed_batch_artifact.path}.csv", index=False, sep=",", quoting=csv.QUOTE_NONE)


@component(base_image='python:3.8', packages_to_install=["google-cloud-aiplatform==1.22.1", "pandas==1.5.3"])
def batch_predict(model_resource_name: str, job_display_name: str, gcs_source: Input[Dataset],
                  predictions_artifact: Output[Artifact], instances_format: str = 'csv',
                  predictions_format: str = 'csv'):
    from google.cloud import aiplatform as vertex_ai

    model = vertex_ai.Model(model_resource_name)
    model.batch_predict(
        job_display_name=job_display_name,
        gcs_source=f'{gcs_source.uri}.csv',
        instances_format=instances_format,
        gcs_destination_prefix=predictions_artifact.uri,
        predictions_format=predictions_format,
        machine_type='n1-standard-8'
    )


@pipeline(pipeline_root=VERTEX_PIPELINES_ROOT, name=PIPELINE_NAME)
def pipeline_fn(project_id: str, region: str, test_size: float, random_state: int, model_display_name: str,
                serving_container: str, label_column: str, predict_job_display_name: str,
                predict_instances_format: str, predictions_format: str, split_column: str = "None"):
    sklearn_to_gcs_op = sklearn_to_gcs().set_display_name("sklearn data to GCS")

    preprocessing_op = preprocessing(
        dataset_artifact=sklearn_to_gcs_op.outputs["dataset_artifact"],
        test_size=test_size,
        random_state=random_state,
    ).set_display_name("preprocessing")

    tfdv_generate_schema_op = tfdv_generate_schema(
        dataset_artifact=preprocessing_op.outputs["dataset_artifact_train"],
        label_column=label_column,
        split_column=split_column,
    ).set_display_name("generate data schema")

    train_op = train(
        dataset_artifact=preprocessing_op.outputs["dataset_artifact_train"],
        label_column=label_column,
    ).set_display_name("training")

    tfdv_validate_with_schema_op = tfdv_validate_with_schema(
        dataset_artifact=preprocessing_op.outputs["dataset_artifact_train"],
        frozen_schema_path=tfdv_generate_schema_op.outputs["schema_artifact"],
    ).set_display_name("test data validation")

    evaluate_op = evaluate(
        model=train_op.outputs["model"],
        dataset_artifact=preprocessing_op.outputs["dataset_artifact_test"],
        label_column=label_column,
    ).set_display_name("evaluation")

    deploy_model_op = deploy_model(
        display_name=model_display_name,
        model=train_op.outputs["model"],
        serving_container=serving_container,
        project_id=project_id,
        location=region,
    ).set_display_name("deployment")

    preprocess_batch_op = preprocess_batch(
        batch_data_gcs=preprocessing_op.outputs["dataset_artifact_inf"]
    ).set_display_name('batch preprocessing')

    batch_predict_op = batch_predict(
        model_resource_name=deploy_model_op.outputs['model_resource_name'],
        job_display_name=predict_job_display_name,
        gcs_source=preprocess_batch_op.outputs['preprocessed_batch_artifact'],
        instances_format=predict_instances_format,
        predictions_format=predictions_format
    ).set_display_name("batch prediction")


if __name__ == "__main__":
    from kfp.v2 import compiler

    compiler.Compiler().compile(
        pipeline_func=pipeline_fn, package_path=f"{PIPELINE_NAME}.json"
    )

    from google.cloud import aiplatform as vertex_ai
    from datetime import datetime

    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    PROJECT_ID = "[PUT-YOUR-PROJECT-ID-HERE]"
    LOCATION = "[PUT-YOUR-REGION-HERE]"

    job = vertex_ai.PipelineJob(
        project=PROJECT_ID,
        location=LOCATION,
        display_name=f"{PIPELINE_NAME}-{TIMESTAMP}",
        template_path=f"{PIPELINE_NAME}.json",
        enable_caching=True,
        parameter_values={
            "project_id": PROJECT_ID,
            "region": LOCATION,
            "test_size": 0.33,
            "random_state": 42,
            "model_display_name": f"wine_xgboost_{TIMESTAMP}",
            "serving_container": "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest",
            "label_column": "target",
            "predict_job_display_name": f"predict_job_{TIMESTAMP}",
            "predict_instances_format": "csv",
            "predictions_format": "jsonl",
            "split_column": "None"
        },
    )

    job.submit()