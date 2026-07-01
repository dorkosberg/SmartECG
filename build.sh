#!/usr/bin/env bash
set -o errexit

echo "Python: $(python --version)"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-web.txt
