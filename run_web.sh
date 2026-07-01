#!/bin/bash
cd "$(dirname "$0")"
echo ""
echo "  SmartECG Assist"
echo "  ==============="
echo ""
if [ ! -f "output/general_model/best_model.pth" ]; then
  echo "  WARNING: Model not found. Run training first:"
  echo "  python src/main.py --mode general-training"
  echo ""
fi
echo "  Starting server at http://127.0.0.1:8080"
echo "  Press Ctrl+C to stop"
echo ""
python web/app.py
