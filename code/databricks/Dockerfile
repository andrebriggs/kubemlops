FROM tensorflow/tensorflow:2.0.0a0-py3

# pip install
RUN pip install databricks-cli && \
    apt-get update && apt-get install jq -y

COPY notebook.py /scripts/notebook.py
COPY run_config.json /scripts/run_config.json
COPY run_notebook.sh /scripts/run_notebook.sh

# will be overwritten by kf pipeline
ENTRYPOINT [ "./scripts/run_notebook.sh" ]
