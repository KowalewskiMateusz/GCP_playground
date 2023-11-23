# Prerequisites


* Choose European Region where all listed API works.
* Listed API should be enabled:
  * Cloud Pub/Sub API
  * Cloud Monitoring API
  * Cloud Logging API
  * Artifact Registry API
  * Secret Manager API
  * Notebooks API
  * Vertex AI API
  * Cloud Functions API
  * Compute Engine API
  * Cloud Composer API
* Create encrypted GCP Artifact Registry Repository.
* Create GCP Service Account which should have possibility to ([IAM basic and predefined roles reference](https://cloud.google.com/iam/docs/understanding-roles)):
  * Work with Vertex AI Services.
  * Read/Write to GCP Artifact Registry Repository.
  * Push/pull Docker images to Artifact Registry.
  * Read and write files to buckets, write logs and metrics to Stackdriver.
  * Encrypt/Decrypt different Cloud Resources via KMS keys.
  * Work with Cloud Functions and data ingestion/preprocessing GCP resources.
* Store GCP Service Account key to **Azure Devops -> Libraries -> Secure files** and use it in **models/azure-pipelines.yml**
* 3 separate **encrypted regional** buckets should be created, public access should be disabled:
  * PIPELINE_ROOT_BUCKET
  * PIPELINE_STORAGE_BUCKET
  * PIPELINE_SHARED_DATA_BUCKET

# Encryption

* [Create separate KMS key](https://cloud.google.com/kms/docs/create-encryption-keys) in GCP Key Management for each GCP resource which you are going to use ( one key for buckets, one key for Artifact Registry,  one key for Composer DAGs and Composer Cluster, one key for all Vertex AI components, etc ).
* Use those keys in CI/CD (**models/azure-pipelines.yml**)

# Data Ingestion

* Create Data Ingestion pipeline via [Cloud Composer](https://cloud.google.com/composer/docs/concepts/overview).
  * [Create Cloud Composer Environment](https://cloud.google.com/composer/docs/how-to/managing/creating).
     * Create **extras/composer_env** folder in **week_4_6** and store there all gcloud commands (canbe stored as **.sh** scripts) which you used to create Cloud Composer Environment.
  * Create and store Cloud Composer DAG file (Data Ingestion Pipeline) to [Cloud Composer Environment Bucket](https://cloud.google.com/composer/docs/composer-2/manage-dags).
     * Create **data_pipeline** folder in **week_4_6** folder and duplicate Cloud Composer DAG file there for version control.
  * (bonus) Optimize Azure DevOps CI/CD pipeline to update DAG in bucket per commit.
* Results should be stored in **PIPELINE_SHARED_DATA_BUCKET** as **.csv** file.
* Data Ingestion pipeline should be triggered once per day via Cloud Composer Cron.

# Preprocessing, Training, Evaluation, Deployment

* Write pipeline components (**model/pipeline_components**) for each pipeline step.
* Write conditional pipeline feature (*with dsl.Condition*) which will execute “deploy_op” (**model/pipeline/end2end.py**) only if the accuracy of the trained model is above the threshold (condition based on “eval_op” (**model/pipeline/end2end.py**) output).
* Compiled Vertex AI Pipeline files (json) should be stored in **PIPELINE_STORAGE_BUCKET.**
* Vertex AI pipeline can be triggered:
  * Locally.
  * Per commit by CI/CD (use **models/azure-pipelines.yml**).
  * (bonus) Each time when files in **PIPELINE_SHARED_DATA_BUCKET** have been updated (Use  Cloud Functions [Storage Trigger](https://cloud.google.com/functions/docs/calling/storage)).
* Get predictions.

# Inference

* Write client code.
* Single request to live model endpoint.
* Stream of requests.
* (bonus) Run batch inference pipeline.
* Create **inference** folder in **week_4_6** and store there all your inference **.py** scripts.
