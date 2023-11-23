from kfp.v2.dsl import component, pipeline
from kfp.v2.dsl import Input, Output, Metrics, Model, Dataset
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
                  dataset_artifact_test: Output[Dataset], test_size: float = 0.33, random_state: int = 42):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(f"{dataset_artifact.path}.csv", sep=",")

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    df_train.to_csv(f"{dataset_artifact_train.path}.csv", index=False, sep=",")
    df_test.to_csv(f"{dataset_artifact_test.path}.csv", index=False, sep=",")


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


@pipeline(pipeline_root=VERTEX_PIPELINES_ROOT, name=PIPELINE_NAME)
def pipeline_fn(project_id: str, region: str, test_size: float, random_state: int,
                model_display_name: str, serving_container: str, label_column: str):
    sklearn_to_gcs_op = sklearn_to_gcs().set_display_name("sklearn data to GCS")

    preprocessing_op = preprocessing(
        dataset_artifact=sklearn_to_gcs_op.outputs["dataset_artifact"],
        test_size=test_size,
        random_state=random_state,
    ).set_display_name("preprocessing")

    train_op = train(
        dataset_artifact=preprocessing_op.outputs["dataset_artifact_train"],
        label_column=label_column,
    ).set_display_name("training")

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
            "serving_container": "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-1:latest",
            "label_column": "target",
        },
    )

    job.submit()