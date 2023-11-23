# Introduction 
Using the following workflow modify exciting Vertex AI pipeline to train an ML model and generate predictions

# Workflow
1. Listed API should be enabled on GCP:
    - Artifact Registry API
    - Notebooks API
    - Vertex AI API
    - Compute Engine API
2. Install dependecies from [requirements](https://dev.azure.com/dsgorg/MLOps-Retrain-Program/_git/gcp-retrain-program?path=/week_1_2/requirements.txt&version=GBmain) to run Vertex AI pipelines
3. Add a new week_3 directory to you GCS bucket and use it as root of new pipelines
4. Add new components to your pipeline from Week 1:
    - add model versioning during performing custom training job
    - create endpoint and deploy your model
5. Investigate [Week 3 Vertex Pipeline](https://dev.azure.com/dsgorg/MLOps-Retrain-Program/_git/gcp-retrain-program?path=/week_3/pipeline_week3.py&version=GBmain):
    - execute it using Cloud Shell or Vertex AI Workbench
    - run and fix any issues appeared
6. Add new components to your pipeline from Week 1. Use as example Week 2 Vertex Pipeline:
    - add generation of data schema component using TFX data validation
    - modify test data validation component and add validation flag as output
    - using KFP dsl condition, run evaluation step only if test data is valid (based on test data validation component)
    - validate inference data with TFX data validation components
    - (bonus) add hyperparameter tuning component
7. Add components for generating predictions:
    - use production data from Week 1-2 to generate batch predictions
    - add component with creating endpoint 
    - add new component for generating online predictions
    - add a dsl condition to generate batch and online predictions only if the new data is valid
8. Run your pipeline and fix any errors appeared 
9. Push the code