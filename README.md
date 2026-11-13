# CarMetric – Used-Car Price Prediction

CarMetric is a machine-learning project that estimates used-car resale prices from vehicle specifications. This version replaces the original manual numeric label mappings and training-set-only model scores with a reproducible preprocessing and evaluation pipeline.

## Stack

Python, Pandas, NumPy, scikit-learn, Streamlit, Joblib

## Project files

- `Car Dataset Processed.csv` – source dataset
- `train_model.py` – data cleaning, preprocessing, cross-validation, held-out testing and model export
- `app.py` – Streamlit prediction interface
- `requirements.txt` – Python dependencies
- `model_evaluation.json` – generated evaluation report
- `final_model.joblib` – generated trained pipeline (created after training)

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Run training before starting the application. The model artifact is generated locally and is not required to be committed if your repository policy excludes binary model files.

## Methodology

- Removes the exported row-index column and duplicate vehicle records.
- Treats insurance validity, fuel type, ownership and transmission as categorical variables using one-hot encoding. This avoids assigning arbitrary numeric order to categories.
- Flags implausible mileage, engine, power and torque values as missing; imputes them within the training pipeline.
- Excludes target prices above ₹500 lakh as likely unit/data-entry anomalies. These records should be checked against the original source before making a final domain decision.
- Compares Random Forest, Extra Trees and HistGradientBoosting using 5-fold cross-validation on the training partition.
- Reports MAE, RMSE and R² on a separate 20% hold-out test partition.
- Selects the final model using cross-validation MAE and refits it on all cleaned records for the app.

## Important limitations

The dataset is small and contains questionable source values, including extreme vehicle measurements and target-price anomalies. The current evaluation is a baseline, not proof of production readiness. Review `model_evaluation.json`, verify questionable source rows, and consider a time-aware or grouped split if the dataset's collection structure requires it. Predicted prices are estimates, not guaranteed market values.

## Academic project

Developed as an academic project by Himanshu Jaiswal.
