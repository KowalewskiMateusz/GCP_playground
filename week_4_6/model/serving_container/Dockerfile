ARG PIPELINE_BASE_IMAGE
FROM ${PIPELINE_BASE_IMAGE}

RUN mkdir -p /app/model
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/

CMD exec gunicorn --bind :$AIP_HTTP_PORT --log-level info --workers 1 --timeout 90 app:app