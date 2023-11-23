import yaml
from typing import NamedTuple


from kfp.v2 import dsl
from kfp import components
from kfp.v2.dsl import (
    Dataset,
    Input,
    Model,
    Output,
    Metrics,
    component,
)

PIPELINE_CONFIG = yaml.safe_load(open("./pipeline_config.yaml"))


PIPELINE_DEFAULT_PARAMS = {
    'model_name': "test_model",
    'serving_container_image_uri': "serving-image",
    "model_endpoint": "model-endpoint",
    'cpu': "8",
    'memory': "32",
    'enable_training': True
}

# This code is a component definition for a Kubeflow pipeline.
# It defines a preprocess component that takes an output dataset as an argument and runs the main() function from the data_preprocessing module.
# The result of the main() function is then written to the path of the output dataset.
# The base image used for this component is specified in the PIPELINE_CONFIG dictionary,
# and it is set to not install the Kubeflow package. 
# The output component file is set to "preprocess.yaml".

@component(
    base_image=PIPELINE_CONFIG["BASE_IMAGE"],
    install_kfp_package=False,
    output_component_file="preprocess.yaml",
)
def preprocess(
        dataset_out: Output[Dataset],
):
        import logging
        from data_preprocessing.main import main

        logging.info(f"begin preprocessing")
        result = main()

        result.to_csv(dataset_out.path, index=False)
        
# This code is a component definition for a Kubeflow Pipeline.
# It defines a component called "train_test_split" which takes in a dataset,
# splits it into two datasets (train and test) based on the test_size parameter, and outputs both datasets.
# The base image for the component is specified in the PIPELINE_CONFIG dictionary,
# and the output component file is set to "train_test_split.yaml".

@component(
    base_image=PIPELINE_CONFIG["BASE_IMAGE"],
    install_kfp_package=False,
    output_component_file="train_test_split.yaml",
)
def train_test_split(dataset_in: Input[Dataset],
                     dataset_train: Output[Dataset],
                     dataset_test: Output[Dataset],
                     test_size: float = 0.2):

    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset_in.path)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=1)

    df_train.to_csv(dataset_train.path)
    df_test.to_csv(dataset_test.path)

# The component takes two arguments, dataset_in and model_out. 
# The types of the arguments are defined in the Input and Output classes.

@component(
    base_image=PIPELINE_CONFIG["BASE_IMAGE"],
    install_kfp_package=False,
    output_component_file="train.yaml",
)
def train(
        dataset_in: Input[Dataset],
        model_out: Output[Model],
):
        from training.main import main
        main(dataset_in.path, model_out.path)

# This code defines a component that evaluates a model on two datasets.
# It takes in two datasets, a model, and outputs two metrics objects.
# The function `eval_dataset` iterates over the metrics returned by the `evaluate` function and logs them.


@component(
    base_image=PIPELINE_CONFIG["BASE_IMAGE"],
    install_kfp_package=False,
    output_component_file="eval.yaml",
)
def eval(
        dataset_train: Input[Dataset],
        dataset_test: Input[Dataset],
        model: Input[Model],
        metrics_train: Output[Metrics],
        metrics_test: Output[Metrics],
):

        from eval.main import evaluate

        def eval_dataset(dataset, metrics):
            for metric_name, val in evaluate(dataset.path, model.path).items():
                metrics.log_metric(metric_name, float(val))

        eval_dataset(dataset_train, metrics_train)
        eval_dataset(dataset_test, metrics_test)


# This code is a component of a pipeline that deploys a trained model to Vertex AI Endpoint.
# It takes in various parameters, such as serving_container_image_uri, service_account, display_name, model_endpoint, model_endpoint_machine_type, gcp_project and gcp_region.
# It then checks if the Vertex AI Endpoint exists and creates it if it does not exist.
# It then uploads the trained model to Vertex AI Model Registry or creates a new model version into an existing uploaded one. Finally, it deploys the trained model to the Vertex AI Endpoint and saves the data to output parameters.

@component(
    base_image=PIPELINE_CONFIG["BASE_IMAGE"],
    install_kfp_package=False,
    output_component_file="deploy.yaml",
)
def deploy(
        serving_container_image_uri: str,
        service_account: str,
        display_name: str,
        model_endpoint: str,
        model_endpoint_machine_type: str,
        gcp_project: str,
        gcp_region: str,
        model: Input[Model],
        vertex_model: Output[Model],
        vertex_endpoint: Output[Model]
        

):
    from google.cloud import aiplatform as vertex_ai
    from pathlib import Path
    
    # Checks existing Vertex AI Enpoint or creates Endpoint if it is not exist.

    def create_endpoint ():
        endpoints = vertex_ai.Endpoint.list(
        filter='display_name="{}"'.format(model_endpoint),
        order_by='create_time desc',
        project=gcp_project,
        location=gcp_region,
        )
        if len(endpoints) > 0:
            endpoint = endpoints[0] # most recently created
        else:
            endpoint = vertex_ai.Endpoint.create(
                display_name=model_endpoint,
                project=gcp_project,
                location=gcp_region,
        )
        return endpoint

    endpoint = create_endpoint()
    
    # Uploads trained model to Vertex AI Model Registry or creates new model version into existing uploaded one.
    
    def upload_model ():
        listed_model = vertex_ai.Model.list(
        filter='display_name="{}"'.format(display_name),
        project=gcp_project,
        location=gcp_region,
        )
        if len(listed_model) > 0:
            model_version = listed_model[0] # most recently created
            model_upload = vertex_ai.Model.upload(
                    display_name=display_name,
                    parent_model=model_version.resource_name,
                    artifact_uri=str(Path(model.path).parent),
                    serving_container_image_uri=serving_container_image_uri,
                    location=gcp_region,
                    serving_container_predict_route="/predict",
                    serving_container_health_route="/health"
            )
        else:
            model_upload = vertex_ai.Model.upload(
                    display_name=display_name,
                    artifact_uri=str(Path(model.path).parent),
                    serving_container_image_uri=serving_container_image_uri,
                    location=gcp_region,
                    serving_container_predict_route="/predict",
                    serving_container_health_route="/health"
            )
        return model_upload
    
    uploaded_model = upload_model()
    
    # Save data to the output params
    vertex_model.uri = uploaded_model.resource_name

    # Deploys trained model to Vertex AI Endpoint
    model_deploy = uploaded_model.deploy(
        machine_type=model_endpoint_machine_type,
        endpoint=endpoint,
        traffic_split={"0": 100},
        deployed_model_display_name=display_name,
        service_account=service_account
    )

    # Save data to the output params
    vertex_endpoint.uri = model_deploy.resource_name


def do_training_condition(do_training: bool) -> str:
    return "true" if do_training else "false"


do_training_condition_op = components.func_to_container_op(do_training_condition, base_image=PIPELINE_CONFIG["BASE_IMAGE"])

# Vertex AI pipeline stages

# This code is defining a pipeline using the dsl library.
# The pipeline takes in parameters such as model_name, serving_container_image_uri, model_endpoint, cpu and memory.
# It also has optional parameters such as test_size and enable_training.
# The pipeline then performs preprocessing, train-test split, training (if enabled), evaluation and deployment operations.
# For each operation, it sets the CPU and memory limits according to the parameters passed in.
# It also uses some configuration values from PIPELINE_CONFIG dictionary for certain operations such as GCP project name,
# GCP region and service account for deployment operation.

@dsl.pipeline(
    name=PIPELINE_CONFIG["PIPELINE_NAME"],
    pipeline_root=PIPELINE_CONFIG["PIPELINE_ROOT"]
)
def pipeline(
         model_name: str,
         serving_container_image_uri: str,
         model_endpoint: str,
         cpu: str,
         memory: str,
         test_size: float = 0.2,
         enable_training: bool = True,
):

    preprocess_op = preprocess().set_cpu_limit(cpu).set_memory_limit(memory)
    
    train_test_split_op = train_test_split(preprocess_op.outputs["dataset_out"], test_size).set_cpu_limit(cpu).set_memory_limit(memory)

    with dsl.Condition(do_training_condition_op(enable_training).output == "true", 
                       name="model-training"):

        train_op = train(train_test_split_op.outputs["dataset_train"]).set_cpu_limit(cpu).set_memory_limit(memory)

        eval_op = eval(train_test_split_op.outputs["dataset_train"],
                       train_test_split_op.outputs["dataset_test"],
                       train_op.outputs["model_out"]
                       ).set_cpu_limit(cpu).set_memory_limit(memory)

        deploy_op = deploy(serving_container_image_uri=serving_container_image_uri,
                           display_name=model_name,
                           gcp_project=PIPELINE_CONFIG["GCP_PROJECT"],
                           gcp_region=PIPELINE_CONFIG["GCP_REGION"],
                           model_endpoint_machine_type=PIPELINE_CONFIG["MODEL_ENDPOINT_MACHINE_TYPE"],
                           model=train_op.outputs["model_out"],
                           model_endpoint=model_endpoint,
                           service_account=PIPELINE_CONFIG["GCP_SERVICE_ACCOUNT"])
