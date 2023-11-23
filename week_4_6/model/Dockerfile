FROM europe-west2-docker.pkg.dev/qbe-eo-ops/vendor-docker/python:3.10-slim-bullseye

RUN apt-get update
RUN apt-get -y install g++

COPY pipeline_components/ ./pipeline_components/


ENV PYTHONPATH "${PYTHONPATH}:/pipeline_components"

RUN pip install --upgrade pip && pip install -r ./pipeline_components/requirements.txt
