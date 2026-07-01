# SmartECG Assist — תיעוד פרויקט מלא

**פרויקט:** ECG Heartbeat Classification  
**מוסד:** Afeka — Machine Learning  
**מודל:** 1D CNN כללי לסיווג בינארי של אות ECG  
**דאטה סט:** MIT-BIH Arrhythmia Database (PhysioNet)  
**תאריך עדכון:** יוני 2026  

---

## תוכן עניינים

1. [סקירת הפרויקט](#1-סקירת-הפרויקט)
2. [מטרה ודרישות](#2-מטרה-ודרישות)
3. [דאטה סט MIT-BIH](#3-דאטה-סט-mit-bih)
4. [עיבוד מקדים (Preprocessing)](#4-עיבוד-מקדים-preprocessing)
5. [חלוקת הנתונים](#5-חלוקת-הנתונים)
6. [ארכיטקטורת המודל — 1D CNN](#6-ארכיטקטורת-המודל--1d-cnn)
7. [אימון והערכה](#7-אימון-והערכה)
8. [תוצאות המודל](#8-תוצאות-המודל)
9. [דף הנחיתה — SmartECG Assist](#9-דף-הנחיתה--smartecg-assist)
10. [מבנה הפרויקט](#10-מבנה-הפרויקט)
11. [פקודות הרצה](#11-פקודות-הרצה)
12. [מסקנות](#12-מסקנות)

---

## 1. סקירת הפרויקט

### 1.1 רקע

הפרויקט פותח במסגרת קורס Machine Learning באפקה. מטרתו לבנות **מסווג בינארי כללי** לאות ECG (אק"ג), שמסווג כל חלון זמן של אות הלב כ:

| תווית | משמעות | ערך |
|-------|--------|-----|
| **Normal** | דופק תקין | 0 |
| **Abnormal** | דופק לא תקין (חריג) | 1 |

### 1.2 שינוי מרכזי בפרויקט

בגרסה המקורית, הפרויקט כלל:
- **Pre-training** על כל המטופלים
- **Fine-tuning** נפרד לכל מטופל
- **מודל נפרד** לכל מטופל (`individuals`)

**בגרסה הנוכחית** (SmartECG Assist):
- **מודל 1D CNN אחד** לכל המטופלים
- **אין** fine-tuning per-patient
- חלוקת דאטה **לפי מזהה מטופל** (מניעת data leakage)
- **דף נחיתה** להעלאת קבצי ECG וקבלת חיזוי בזמן אמת

### 1.3 זרימת המערכת

```
MIT-BIH CSV + Annotations
        ↓
   Preprocessing
   (סינון, נרמול, חלונות, resample)
        ↓
   חלוקה: Train / Val / Test (לפי מטופל)
        ↓
   אימון 1D CNN
        ↓
   שמירת best_model.pth
        ↓
   הערכה על מטופלי Test
        ↓
   דף נחיתה (SmartECG Assist) — העלאת CSV חדש
```

---

## 2. מטרה ודרישות

| # | דרישה | מימוש |
|---|--------|--------|
| 1 | סיווג בינארי — תקין / לא תקין | תוויות 0/1, BCELoss, Sigmoid |
| 2 | Preprocessing: סינון, נרמול, חלונות | `data_processing.py` |
| 3 | דאטה סט MIT-BIH | `dataset/mitbih_database/` |
| 4 | חלוקה Train / Validation / Test | `split_patients_by_id()` |
| 5 | 1D CNN | `models/cnn1D.py` |

---

## 3. דאטה סט MIT-BIH

### 3.1 מקור

- **שם:** MIT-BIH Arrhythmia Database  
- **קישור:** https://www.physionet.org/content/mitdb/1.0.0/  
- **נתיב בפרויקט:** `dataset/mitbih_database/`

### 3.2 מבנה הקבצים

לכל מטופל (record) יש שני קבצים:

| קובץ | תוכן |
|------|------|
| `{id}.csv` | אות ECG — עמודות: sample #, MLII, V5 (לעיתים) |
| `{id}annotations.txt` | תיוג beat-by-beat |

**דוגמה** (`100.csv`):
```
'sample #','MLII','V5'
0,995,1011
1,995,1011
...
```

### 3.3 פרמטרים טכניים

| פרמטר | ערך |
|--------|-----|
| קצב דגימה | 360 Hz |
| מספר מטופלים בדאטה | 48 |
| מטופלים לאחר סינון | 29 |
| משך רשומה טיפוסי | ~30 דקות |

### 3.4 תיוגים (Annotations)

**Normal (0):**
- `N` — Normal beat

**Abnormal (1):**
- `L`, `R`, `V`, `A`, `a`, `j`, `S`, `F`, `E`, `e`, `r`, `!`, `f`, `/`

**נזרקים:**
- `Q`, `?` — לא מסווגים

### 3.5 סינון מטופלים

מטופלים עם פחות מ-**0.9%** דופקים abnormal **מוצאים** מהאימון (חוסר איזון קיצוני).

**19 מטופלים הוצאו:**  
101, 103, 107, 109, 111, 112, 113, 115, 117, 118, 121, 122, 123, 124, 207, 214, 230, 232, 234

---

## 4. עיבוד מקדים (Preprocessing)

כל עיבוד מתבצע ב-`src/data_processing.py` (אימון) ו-`src/inference.py` (דף נחיתה).

### 4.1 שלב 1 — טעינה

- קריאת אות ECG מהעמודה השנייה (Lead MLII)
- קריאת annotations לתיוג (באימון בלבד)

### 4.2 שלב 2 — סינון רעש (Bandpass Filter)

| פרמטר | ערך |
|--------|-----|
| סוג | Butterworth Bandpass |
| תדר נמוך | 0.4 Hz |
| תדר גבוה | 30 Hz |
| קצב דגימה | 360 Hz |
| סדר (order) | 3 |

**מטרה:** הסרת baseline wander ורעשים בתדרים גבוהים.

### 4.3 שלב 3 — Normalization

- **Min-Max normalization** לטווח [0, 1] **לכל מטופל בנפרד**
- נוסחה: `(signal - min) / (max - min)`

### 4.4 שלב 4 — Segmentation (חלונות זמן)

- חלוקה לחלונות של **שנייה אחת** = **360 דגימות** (ב-360 Hz)
- לכל חלון — תיוג לפי annotations:
  - אם יש לפחות beat abnormal אחד בחלון → **Abnormal (1)**
  - אחרת, אם יש Normal → **Normal (0)**
  - beats לא מסווגים → החלון **נזרק**

### 4.5 שלב 5 — Resampling

- כל חלון מושם לגודל קבוע: **128 דגימות**
- נדרש כדי שכל קלט ל-CNN יהיה באורך אחיד

### 4.6 שלב 6 — Reshape

- המרה מ-`(N, 128)` ל-`(N, 1, 128)` — ערוץ אחד, אורך 128

---

## 5. חלוקת הנתונים

### 5.1 עקרון מרכזי

החלוקה מתבצעת **לפי מזהה מטופל (Patient ID)** — **לא** אקראית לפי beat.

> כל ה-beats של מטופל נמצאים **רק** באחד מה-splits.  
> זה מונע **data leakage** ומאפשר לבדוק התכללות למטופלים חדשים.

### 5.2 יחסי החלוקה

| Split | אחוז | מספר מטופלים |
|-------|------|-------------|
| **Train** | ~65.5% | 19 |
| **Validation** | ~17.2% | 5 |
| **Test** | ~17.2% | 5 |

- **Seed:** 42 (לשחזוריות)
- **פרמטרים:** `val_ratio=0.15`, `test_ratio=0.15`

### 5.3 רשימת מטופלים

**Train (19):**  
100, 105, 106, 108, 114, 116, 201, 202, 205, 208, 209, 212, 215, 219, 221, 222, 223, 228, 233

**Validation (5):**  
102, 104, 200, 213, 217

**Test (5):**  
119, 203, 210, 220, 231

החלוקה נשמרת ב: `output/general_model/patient_splits.json`

---

## 6. ארכיטקטורת המודל — 1D CNN

המודל מוגדר ב-`src/models/cnn1D.py` — רשת CNN חד-ממדית בסגנון ResNet (skip connections).

### 6.1 קלט ופלט

| פריט | ערך |
|------|-----|
| **Input** | `(batch, 1, 128)` — חלון ECG בודד |
| **Output** | הסתברות בין 0 ל-1 (Sigmoid) |
| **Threshold** | ≥ 0.5 → Abnormal, < 0.5 → Normal |

### 6.2 מבנה הרשת

```
Input (1, 128)
    ↓
Conv1d: 1 → 32 channels, kernel=1
    ↓
ReLU
    ↓
4 × Block:
    Conv1d → Conv1d → Skip Connection → ReLU → MaxPool1d
    (32 channels, kernel=5)
    ↓
Flatten → 160 features
    ↓
Linear: 160 → 160
    ↓
Linear: 160 → 1
    ↓
Sigmoid → הסתברות
```

### 6.3 Block (בלוק בסיסי)

כל Block כולל:
1. **Conv1d** — שכבת קונבולוציה ראשונה + ReLU
2. **Conv1d** — שכבת קונבולוציה שנייה
3. **Skip connection** — חיבור קצר (residual)
4. **MaxPool1d** — הקטנת ממדים

### 6.4 פרמטרים

| פרמטר | ערך ברירת מחדל |
|--------|----------------|
| `num_blocks` | 4 |
| `block_channels` | 32 |
| `kernel_size` | 5 |
| `num_features` | 160 |

### 6.5 פונקציית Loss

- **Binary Cross Entropy (BCELoss)**
- Optimizer: **Adam**, lr=0.001
- LR Scheduler: **ReduceLROnPlateau**

---

## 7. אימון והערכה

### 7.1 הגדרות אימון

| פרמטר | ערך |
|--------|-----|
| Epochs | 30 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Weighted sampling | True |
| Device | CPU / CUDA |

### 7.2 בחירת המודל

- נשמר **`best_model.pth`** לפי **validation loss הנמוך ביותר**
- המודל שנבחר: **Epoch 5** (val loss = 0.4714)
- לאחר האימון — טעינת best model והערכה על **מטופלי Test**

### 7.3 פקודות

**אימון:**
```bash
python src/main.py --mode general-training --output_path ./output/general_model/
```

**הערכה:**
```bash
python src/main.py --mode test --model_path ./output/general_model/best_model.pth
```

---

## 8. תוצאות המודל

### 8.1 מטריקות על Test Set (מטופלים חדשים)

| מטריקה | ערך |
|--------|-----|
| **Accuracy** | 84.07% |
| **Precision** | 0.639 |
| **Recall** | 0.933 |
| **F1-score** | 0.758 |
| **Balanced Accuracy** | 86.98% |

### 8.2 Confusion Matrix

```
                    חיזוי Normal    חיזוי Abnormal
בפועל Normal          5,064 (TN)       1,211 (FP)
בפועל Abnormal          155 (FN)       2,144 (TP)
```

### 8.3 פרשנות

- **Recall גבוה (93%)** — המודל תופס את רוב הדופקים החריגים
- **Precision בינוני (64%)** — יש false positives (תקינים שסווגו כחריגים)
- **Overfitting** — Train accuracy ~99% לעומת Test ~84%

---

## 9. דף הנחיתה — SmartECG Assist

### 9.1 מטרה

ממשק ווב לבדיקת אות ECG חדש **ללא צורך ב-annotations** — המשתמש מעלה קובץ CSV ומקבל:
- תוצאה: **תקין / לא תקין**
- **מידת ביטחון** (אחוזים)
- **גרף אות ECG** עם סימון חלונות abnormal
- סטטיסטיקות (חלונות שנותחו, עמודה שזוהתה)

### 9.2 קבצים

| קובץ | תפקיד |
|------|--------|
| `web/app.py` | שרת Flask |
| `web/templates/index.html` | ממשק משתמש |
| `src/inference.py` | טעינה, preprocessing, חיזוי, גרף |
| `SmartECG_Assist.command` | הפעלה בלחיצה כפולה (Mac) |

### 9.3 הפעלה

**דרך 1 — לחיצה כפולה:**
```
SmartECG_Assist.command
```

**דרך 2 — טרמינל:**
```bash
cd ECG-Heartbeat-Classification
python web/app.py
```
ואז: **http://127.0.0.1:8080**

> **חשוב:** אין לפתוח את `index.html` ישירות מהתיקייה — חובה דרך השרת.

### 9.4 תמיכה בפורמטי קבצים

| פורמט | נתמך |
|--------|--------|
| MIT-BIH (`sample #, MLII, V5`) | ✅ |
| עמודה אחת עם ערכי ECG | ✅ |
| CSV עם מפריד `;` או TAB | ✅ |
| קבצי `.txt`, `.tsv` | ✅ |
| זיהוי אוטומטי של עמודת ECG | ✅ |

### 9.5 קצב דגימה

- ברירת מחדל: **360 Hz** (MIT-BIH)
- ניתן לשנות בשדה "קצב דגימה (Hz)" בדף
- האות מנורמל ל-360 Hz לפני הניתוח

### 9.6 חישוב תוצאות בדף

1. האות מתחלק לחלונות של שנייה (360 דגימות)
2. המודל מחזיר הסתברות abnormal לכל חלון (עד 300 חלונות)
3. **תוצאה כוללת:** ממוצע ההסתברויות ≥ 0.5 → Abnormal
4. **ביטחון:** אם Abnormal → הממוצע; אם Normal → 1 − ממוצע
5. **גרף:** 15 השניות הראשונות, אזורים אדומים = חלונות abnormal

### 9.7 מגבלות

- נותחים עד **300 חלונות** (~5 דקות) לביצועים
- המודל אומן על MIT-BIH — תוצאות על דאטה סטים אחרים עשויות להיות פחות מדויקות
- אין אבחנה רפואית — כלי מחקר/הדגמה בלבד

---

## 10. מבנה הפרויקט

```
ECG-Heartbeat-Classification/
├── dataset/mitbih_database/     # דאטה סט MIT-BIH
├── src/
│   ├── main.py                  # נקודת כניסה — אימון / test
│   ├── data_processing.py       # preprocessing + חלוקה
│   ├── inference.py             # חיזוי לדף נחיתה
│   ├── running.py               # לולאת אימון
│   ├── models/cnn1D.py          # ארכיטקטורת 1D CNN
│   └── training/                # train / val / test epochs
├── web/
│   ├── app.py                   # שרת Flask
│   └── templates/index.html     # SmartECG Assist UI
├── output/general_model/
│   ├── best_model.pth           # מודל שנבחר
│   ├── patient_splits.json      # חלוקת מטופלים
│   └── *.log                    # לוגים
├── docs/                        # מסמכי תיעוד
├── SmartECG_Assist.command        # הפעלת דף נחיתה
└── run_web.sh
```

---

## 11. פקודות הרצה

| פעולה | פקודה |
|--------|--------|
| אימון מודל | `python src/main.py --mode general-training` |
| הערכת מודל | `python src/main.py --mode test --model_path ./output/general_model/best_model.pth` |
| דף נחיתה | `python web/app.py` → http://127.0.0.1:8080 |
| הפעלה מהירה (Mac) | לחיצה כפולה על `SmartECG_Assist.command` |

---

## 12. מסקנות

1. הפרויקט מממש **מסווג ECG בינארי כללי** על MIT-BIH עם **1D CNN**.
2. **חלוקה לפי מטופל** מבטיחה הערכה הוגנת על מטופלים חדשים.
3. המודל משיג **84% accuracy** ו-**93% recall** על מטופלי test.
4. **SmartECG Assist** מאפשר בדיקת קבצי ECG חדשים דרך ממשק ווב.
5. יש מקום לשיפור ב-precision ולהפחתת overfitting.

---

*מסמך זה נוצר עבור פרויקט ECG Heartbeat Classification — Afeka, Machine Learning.*
