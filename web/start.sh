#!/bin/bash
cd "$(dirname "$0")/.."
echo "Starting ECG Classifier Web UI..."
python web/app.py
