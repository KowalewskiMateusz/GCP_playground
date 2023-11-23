import os
import sys
import json
import re
import logging
from typing import Tuple, List, Dict

import pandas as pd
from flask import Flask, request, abort
from google.cloud import storage

from predict.main import load_model, model_predict

logger = logging.getLogger("App")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

app = Flask(__name__)


SERVER_PORT = os.environ.get("AIP_HTTP_PORT", 8080)
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_STORAGE_URI = os.environ["AIP_STORAGE_URI"]  # Vertex AI sets this env with path to the model artifact
logger.info(f"MODEL PATH: {AIP_STORAGE_URI}")

model_path = "model/model.pickle"  # Local path
DATAFRAME_NAME = "df.csv"

# This code takes a Google Cloud Storage URL (e.g. gs://bucket_name/dir1/filename)
# and splits it into two parts:
# the bucket name (e.g. bucket_name)
# and the path after the bucket (e.g. dir1/filename).
# It uses regular expressions to find the bucket name, 
# then uses string splitting to get the path after the bucket.
# It returns both parts as a tuple of strings.

def decode_gcs_url(url: str) -> Tuple[str, str, str]:
    """
        Split a google cloud storage path such as: gs://bucket_name/dir1/filename into
        bucket and path after the bucket: bucket_name, dir1/filename
        :param url: storage url
        :return: bucket_name, blob
        """
    bucket = re.findall(r'gs://([^/]+)', url)[0]
    blob = url.split('/', 3)[-1]
    return bucket, blob

# This function uploads a file from a local path to Google Cloud Storage. 
# The function takes in two arguments: local_path and remote_path. 
# The function then creates a storage client and decodes the remote path into a bucket and blob. 
# The function then creates a bucket and blob, 
# uploads the file from the local path to the remote path, 
# and returns the public url of the uploaded file.

def gcs_upload(local_path: str, remote_path: str):
    storage_client = storage.Client()
    dst_bucket, dst_blob = decode_gcs_url(remote_path)
    bucket = storage_client.bucket(dst_bucket)
    blob = bucket.blob(dst_blob)
    blob.upload_from_filename(local_path)

# This function takes two parameters: an artifacts URI (artifacts_uri) and a local path (local_path).
# It then uses the storage.Client() method to create a storage client, 
# which is used to access the bucket and blob associated with the artifacts URI.
# Finally, it downloads the file from the artifacts URI to the local path.

def download_artifacts(artifacts_uri: str, local_path: str):
    logger.info(f"Downloading {artifacts_uri} to {local_path}")
    storage_client = storage.Client()
    src_bucket, src_blob = decode_gcs_url(artifacts_uri)
    source_bucket = storage_client.bucket(src_bucket)
    source_blob = source_bucket.blob(src_blob)
    source_blob.download_to_filename(local_path)
    logger.info(f"Downloaded.")

# This function loads artifacts from GCS by downloading them into
# local storage of service container using gcs_download function call with
# model path and model name as parameters respectively.

def load_artifacts(artifacts_uri: str = AIP_STORAGE_URI):
    model_uri = os.path.join(artifacts_uri, "model_out")
    logger.info(f"Loading artifacts from {model_uri}")
    download_artifacts(model_uri, model_path)


@app.route(HEALTH_ROUTE, methods=["GET"])
def health_check():
    return "I am alive", 200

# This code is the serving endpoint of the service container.
# It takes in a list of instances, which can be either a list of instances or a file path to a csv file.
# If it's a file path, it will download the file from GCS and convert it into a dataframe. 
# Then it will run model_predict on the dataframe to get the predictions. 
# Finally, if the input is a file path, it will upload the scored dataframe into GCS and return that file path as output. 
# Otherwise, it will return the predictions as output.

@app.route(PREDICT_ROUTE, methods=["POST"])
def predict():
    """
    Predict endpoint of service container.
    By Google spec input is a list of "instances"
    """
    logger.info("SERVING ENDPOINT: Received predict request.")  # do not change this line    
    
    # bootstrap section: load model(s), extract and understand payload
    if not os.path.exists(model_path):
        logger.info(f"No model object. Caching from {model_path}")
        load_artifacts()
    model = load_model(model_path)
    logger.info(f"Model loaded")
    payload = json.loads(request.data)

    instances = payload["instances"]
    file_path = instances[0]

    logger.debug(f"Payload: {payload}, type: {type(payload)}")
    logger.debug(f"Instances: {instances}, type: {type(instances)}")
    logger.debug(f"file_path: {file_path}, type: {type(file_path)}")

    data_in_file = len(instances) == 1 and "gs:/" in file_path

    # predict section: run model scoring
    if data_in_file:
        try:
            logger.info("Received file scoring request, downloading...")
            download_artifacts(file_path, DATAFRAME_NAME)
            instances = pd.read_csv(DATAFRAME_NAME)
        except Exception as e:
            logger.error(f"Failed to process payload:\n {e}")
            abort(500, "Failed to score request.")
    else:
        logger.info("Received instances lists, converting...")
        df_str = "\n".join(instances)
        instances = pd.read_json(df_str, lines=True)
    
    logger.info("Running MODEL_PREDICT for request.")
    model_output, target_name = model_predict(model, instances)
    logger.info("MODEL_PREDICT completed.")
    
    # response section: package data and send back
    if data_in_file:
        instances[target_name] = model_output
        # persist the results
        instances.to_csv(DATAFRAME_NAME, header=True, index=False)
        output_name = f"{file_path.split('.')[0]}_scored.csv"  # remote filename
        logger.info(f"Uploading scores to {output_name}")
        gcs_upload(DATAFRAME_NAME, output_name)
        response = {"predictions": [output_name]}
    else:
        response = {"predictions": model_output.tolist()}
    logger.info("SERVING ENDPOINT: Finished processing.")  # do not change this line    
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
