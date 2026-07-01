#!/bin/bash
cd "$(dirname "$0")"

PORT=8080
URL="http://127.0.0.1:${PORT}"

echo ""
echo "  SmartECG Assist"
echo "  ==============="
echo ""

if [ ! -f "output/general_model/best_model.pth" ]; then
  echo "  שגיאה: המודל לא נמצא."
  echo "  הרץ קודם: python src/main.py --mode general-training"
  echo ""
  read -p "לחץ Enter לסגירה..."
  exit 1
fi

# אם השרת כבר רץ — רק פותחים דפדפן
if curl -s -m 1 "${URL}/health" >/dev/null 2>&1; then
  echo "  השרת כבר פעיל. פותח דפדפן..."
  open "${URL}"
  read -p "לחץ Enter לסגירה..."
  exit 0
fi

echo "  מפעיל שרת על ${URL}"
echo "  אל תסגור את החלון הזה בזמן השימוש!"
echo ""

export PORT="${PORT}"
python web/app.py &
SERVER_PID=$!
sleep 2

if curl -s -m 2 "${URL}/health" >/dev/null 2>&1; then
  open "${URL}"
  echo "  הדפדפן נפתח. לעצירה: Ctrl+C"
  wait ${SERVER_PID}
else
  echo "  לא הצלחתי להפעיל את השרת."
  kill ${SERVER_PID} 2>/dev/null
  read -p "לחץ Enter לסגירה..."
  exit 1
fi
