#!/bin/bash
docker load -i face_lvt.tar
docker-compose -f docker-compose.yml up -d