# Introduction 
This repository defines a Vertex AI pipeline and a custom serving container.


# Pipeline

The Pipeline is designed as a generic machine learning pipeline that is intended to be used as a template
for multiple machine learning models that can reuse its basic flow:

**data preprocessing -> train/test split -> training -> evaluate**

The behaviour of these steps is defined in the *pipeline_components* directory which is in the PYTHONPATH of the base image for each pipeline step. A main function from each step is called inside each pipeline step(component).

### Build and Test

The Vertex AI pipeline and its requirements are defined in the files under the *pipeline* directory.


# Serving Container

The serving container is a simple flask app confronting to [requirements](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements) set by the Vertex AI platform.

# CI/CD

This repo is part of A CI/CD pipeline on Azure Devops. It is defined in *azure-pipelines.yaml*
