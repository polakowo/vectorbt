#!/bin/bash

docker tag "${1}" "${1}:${2}"
docker tag "${1}" "${1}:latest"
docker push "${1}" --all-tags