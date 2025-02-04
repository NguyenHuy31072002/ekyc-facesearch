#!/bin/bash
if [ "$1" == "docker" ]
then
  echo "========================= TF SERVING DOCKER ========================="
  docker run -d -p 8501:8501 -p 8500:8500  --name ImageModels \
      --restart always \
      --mount type=bind,source="$(pwd)"/models/,target=/models/  \
      --mount type=bind,source="$(pwd)"/config/,target=/config/  \
      -t tensorflow/serving \
      --model_config_file=/config/docker_serve.config \
      --model_config_file_poll_wait_seconds=60 \
      --monitoring_config_file=/config/prometheus.config
elif [ "$1" == "docker-gpu" ]
then
  echo "========================= TF SERVING DOCKER GPU ========================="
  docker run -d --runtime=nvidia -p 8501:8501 -p 8500:8500  --name ImageModels \
      --restart always \
      --mount type=bind,source="$(pwd)"/models/,target=/models/  \
      --mount type=bind,source="$(pwd)"/config/,target=/config/  \
      -t tensorflow/serving:latest-gpu \
      --model_config_file=/config/docker_serve.config \
      --model_config_file_poll_wait_seconds=60 \
      --monitoring_config_file=/config/prometheus.config
elif [ "$1" == "server" ]
then
  echo "========================= TF SERVING SERVER ========================="
  tensorflow_model_server --port=8500 \
      --rest_api_port=8501 \
      --model_config_file="$(pwd)"/config/serve.config 
fi
