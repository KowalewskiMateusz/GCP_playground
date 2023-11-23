import os
import argparse
from datetime import datetime

from kfp.v2 import compiler
from google.cloud import aiplatform as vertex_ai
from google.oauth2 import service_account

from pipeline.end2end import pipeline
from pipeline.end2end import  PIPELINE_DEFAULT_PARAMS

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
GCP_SERVICE_ACCOUNT = os.getenv("GCP_SERVICE_ACCOUNT")
GCP_SA_KEY_PATH = os.getenv("authkey.secureFilePath")
PIPELINE_NAME = os.getenv("MODEL_NAME")

PIPELINE_DEFAULT_PARAMS.update({
    "model_name": os.environ.get("MODEL_NAME"),
    "serving_container_image_uri": os.environ.get("PIPELINE_SERVING_IMAGE"),
    "model_endpoint": os.environ.get("MODEL_ENDPOINT"),
    "cpu": os.environ.get("CPU"),
    "memory": os.environ.get("MEMORY")
})

# This code is used to deploy a pipeline to Google Cloud Platform (GCP).
# The first block of code parses command line arguments,
# such as the path of the pipeline package and credentials.
 
# The second block of code compiles the pipeline and exports it to the specified package path.

# The third block of code creates a pipeline job with a unique job ID,
# sets up credentials, and submits the job to GCP.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--package_path", default="pipeline.json")
    parser.add_argument("--credentials_path", default="key.json")

    args = parser.parse_args()

    compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=args.package_path,
    pipeline_parameters=PIPELINE_DEFAULT_PARAMS
    )
    print(f"successfully exported pipeline to {args.package_path}")

    if args.deploy:
        TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # this way we do not need to set GOOGLE_APPLICATION_CREDENTIALS environmental variable
        credentials = service_account.Credentials.from_service_account_file(args.credentials_path)
        
        pipeline_job = vertex_ai.PipelineJob(
            display_name=PIPELINE_NAME,
            template_path=args.package_path,
            job_id=f"{PIPELINE_NAME}-{TIMESTAMP}",
            enable_caching=False,
            project=GCP_PROJECT,
            location=GCP_REGION,
            credentials=credentials
        )

        pipeline_job.submit(service_account=GCP_SERVICE_ACCOUNT)
