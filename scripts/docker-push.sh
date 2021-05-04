#!/bin/bash

TARGET_IMAGE_LATEST="${SOURCE_IMAGE}:latest"
TARGET_IMAGE_TAGGED="${SOURCE_IMAGE}:${IMAGE_TAG}"

# push new version
docker tag "${SOURCE_IMAGE}" "${TARGET_IMAGE_TAGGED}"
docker push "${TARGET_IMAGE_TAGGED}"

# update latest version
docker tag "${SOURCE_IMAGE}" "${TARGET_IMAGE_LATEST}"
docker push "${TARGET_IMAGE_LATEST}"