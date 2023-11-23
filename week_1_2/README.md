# Introduction 
Using the following workflow create a Vertex AI pipeline to train an ML model

# Workflow
1. Listed API should be enabled on GCP:
    - Artifact Registry API
    - Notebooks API
    - Vertex AI API
    - Compute Engine API
    - Cloud Storage API
2. Install dependecies from [requirements](https://dev.azure.com/dsgorg/MLOps-Retrain-Program/_git/gcp-retrain-program?path=/week_1_2/requirements.txt&version=GBmain) to run Vertex AI pipelines
3. Select region that will work better for you. It also should be applicable with Vertex AI services.
4. Create a GCS bucket for use as a pipeline root and data storage. Be careful when choosing bucket preferences.
    - naming convention: mlops-retraining-[your_username]
    - add a directory inside: week_1_2
5. Select a new tabular or timeseries dataset to run the pipeline.:
    - split data into training/testing (90%) and production data (10%)
    - save this data to GCS bucket
6. Investigate [Week 1-2 Vertex Pipeline](https://dev.azure.com/dsgorg/MLOps-Retrain-Program/_git/gcp-retrain-program?path=/week_1_2/pipeline_week_1_2.py&version=GBmain):
    - execute it using Cloud Shell or Vertex AI Workbench
    - run and fix any issues appeared
7. Select a new tabular or timeseries dataset to run the pipeline. You can use public BiqQuery datasets or any other data of your choice (data should include numeric and categorical variables).
8. Pipeline refactoring:
    - select a new tabular or timeseries dataset to run the pipeline. You can use public BiqQuery datasets or any other data of your choice (data should include numeric and categorical variables).
    - add a component to create a managed Vertex AI dataset from BQ or GCS based on your choice instead of sklearn_to_gcs.
    - develop a sklearn training module for your dataset. Develop preprocessing trasnformers and training code and  wrap it with the sklearn pipeline.
    - select serving container from prebuilt containers and put it to pipeline parameters. **Make sure that you select component without GPU.**
    - add a training component with creating and running CustomTrainingJob or CustomPythonPackageTrainingJob. You must choose one of them based on your training code. Describe in comments why you made this decision. Vertex AI Custom Training automatically deploys model to Vertex AI. So now you could remove deploy_model component.
    - add export model component and store model artifact on GCS bucket.
    - modify evaluation component if needed and update metrics.
    - (bonus) add Vertex AI Feature Store component to your pipeline
9. Run your pipeline and fix any errors appeared 
10. Push the code