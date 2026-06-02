# ❤️ Heart Disease Prediction Dashboard
### WQD7001 – Principles of Data Science | Group Project

A Streamlit web application that uses a trained XGBoost model to predict heart disease risk based on patient health indicators.

---

## 📁 Required Files

Place these files in the **same folder** as `app.py`:

```
heart_disease_app/
├── app.py
├── requirements.txt
├── best_xgboost_model.json   ← from your notebook output
└── README.md
```

> **Note:** The `best_xgboost_model.json` was exported from your notebook using `model.save_model('best_xgboost_model.json')`.

---

## 🚀 Running Locally

### Step 1 – Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### Step 2 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 – Run the app
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Cloud (Free)

1. Push your project folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo, branch, and `app.py`
4. Click **Deploy** — it will be live in ~2 minutes at a public URL

---

## 📊 Dashboard Features

| Feature | Description |
|---|---|
| **Sidebar Inputs** | 18 patient health indicators |
| **Prediction Result** | Binary prediction + probability percentage |
| **Risk Gauge** | Colour-coded dial (green / amber / red) |
| **Risk Flags** | 9 clinical risk factor indicators |
| **Feature Importance** | Top 15 XGBoost feature importances |
| **Patient Radar Chart** | Normalised multi-axis risk profile |
| **Model Comparison Table** | LR / DT / RF / XGBoost metrics |
| **Model & Dataset Info** | Architecture and dataset summary cards |

---

## ⚠️ Scaler Note

The standard scaler used during training was **not saved** in the notebook.  
To ensure exact predictions, add this line to your notebook after fitting the scaler:

```python
import joblib
joblib.dump(scaler, 'scaler.pkl')
```

Then in `app.py`, replace the `SCALER_MEAN` / `SCALER_STD` dictionaries with:

```python
import joblib
scaler = joblib.load('scaler.pkl')
# Apply in preprocess():
row_df[num_cols] = scaler.transform(row_df[num_cols])
```

---

## 👥 Group Project
**Course:** WQD7001 – Principles of Data Science  
**Title:** Heart Disease Prediction Using Machine Learning: A Study of Model Effectiveness and Feature Significance  
**Deployment Section:** Data Product (Item 7)
