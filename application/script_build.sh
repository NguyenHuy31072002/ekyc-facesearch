#!/bin/bash
docker build -t face_lvt:v2 .
docker save -o face_lvt.tar face_lvt:v2
#docker push face_lvt:latest