#!/bin/bash
set -e
cd "$(dirname "$0")/.."

rm -rf dist/ build/ *.egg-info
python3 -m build
twine upload dist/*
