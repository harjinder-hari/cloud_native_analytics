#!/usr/bin/env bash

# --------------------------------------------------------------------------------------------------
# start web service to provide rest end points for this container
# --------------------------------------------------------------------------------------------------
export SPARK_HOME=/spark-2.0.2-bin-hadoop2.7
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.3-src.zip
gunicorn --pythonpath / -b 0.0.0.0:$SERVICE_PORT -t $SERVICE_TIMEOUT scoring_service:app

tail -f /scoring_service.py
