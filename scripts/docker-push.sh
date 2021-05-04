#!/bin/bash

TARGET_IMAGE_TAGGED="${1}:${2}"
TARGET_IMAGE_LATEST="${1}:latest"

docker tag "${1}" "${TARGET_IMAGE_TAGGED}"
docker push "${TARGET_IMAGE_TAGGED}"

docker tag "${1}" "${TARGET_IMAGE_LATEST}"
docker push "${TARGET_IMAGE_LATEST}"