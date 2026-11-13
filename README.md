# 🚗 CarMetric – Used Car Price Prediction System

<p align="center">
  <strong>An End-to-End Machine Learning Application for Predicting Used Car Resale Prices</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Data%20Analysis-Pandas-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 📌 Project Overview

**CarMetric** is a machine learning-based used car price prediction system developed using Python. It estimates the resale value of a used vehicle based on its specifications, condition, ownership information, and other relevant attributes.

The project implements an end-to-end machine learning workflow, covering data preprocessing, exploratory preparation, feature encoding, model training, cross-validation, and model evaluation.

An interactive web application built with **Streamlit** allows users to enter vehicle details and receive estimated resale prices through a trained machine learning model.

The primary objective is to demonstrate the practical application of Data Science and Machine Learning in solving a real-world regression problem.

## ✨ Key Features

- **Used Car Price Prediction:** Estimates vehicle resale prices using a trained regression model.
- **Interactive Web Interface:** Provides an easy-to-use interface powered by Streamlit.
- **Data Preprocessing:** Handles duplicate records, missing or invalid values, and data quality checks.
- **Feature Engineering:** Processes numerical and categorical vehicle attributes.
- **Categorical Encoding:** Uses One-Hot Encoding to represent categorical features.
- **Model Comparison:** Evaluates multiple regression algorithms to identify a suitable predictive model.
- **Cross-Validation:** Uses 5-fold cross-validation to assess model performance.
- **Model Persistence:** Saves the trained machine learning pipeline using Joblib for reuse.
- **Performance Evaluation:** Measures prediction quality using MAE, RMSE, and R².

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Regression Models | Random Forest, Extra Trees, HistGradientBoosting |
| Data Preprocessing | One-Hot Encoding, Feature Validation |
| Model Evaluation | MAE, RMSE, R², Cross-Validation |
| Model Persistence | Joblib |
| Web Application | Streamlit |
| Development Tools | VS Code, Jupyter Notebook, Git, GitHub |

## 🧠 Machine Learning Workflow

CarMetric follows a structured machine learning pipeline:

1. **Data Collection:** Loads the used car dataset from a CSV file.
2. **Data Cleaning:** Identifies duplicate records and checks data quality.
3. **Data Preprocessing:** Handles categorical and numerical features and validates vehicle attributes.
4. **Feature Encoding:** Applies One-Hot Encoding to categorical variables.
5. **Train-Test Split:** Divides the dataset into training and testing subsets using an 80:20 split.
6. **Model Training:** Trains and compares multiple regression algorithms.
7. **Cross-Validation:** Uses 5-fold cross-validation to assess model consistency.
8. **Model Evaluation:** Compares models using MAE, RMSE, and R² metrics.
9. **Model Saving:** Exports the selected trained pipeline using Joblib.
10. **Deployment:** Integrates the saved model into a Streamlit application for interactive predictions.

## 🤖 Models Evaluated

The project compares the following regression algorithms:

| Algorithm | Description |
|---|---|
| Random Forest Regressor | Ensemble learning method using multiple decision trees |
| Extra Trees Regressor | Randomized tree-based ensemble learning algorithm |
| HistGradientBoosting Regressor | Gradient boosting algorithm designed for efficient training |

The model selection process considers cross-validation performance, particularly Mean Absolute Error (MAE), alongside held-out test-set evaluation.

## 📊 Model Evaluation

The following results were obtained during model evaluation:

| Model | Test MAE (₹ Lakh) | Test RMSE (₹ Lakh) | Test R² |
|---|---:|---:|---:|
| Random Forest | 4.30 | 10.52 | 0.591 |
| Extra Trees | 4.52 | 10.52 | 0.590 |
| HistGradientBoosting | 4.77 | 10.28 | 0.609 |

**Evaluation metrics:**

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.
- **R² Score:** Measures the proportion of target-price variance explained by the model.

**Model Selection:** Random Forest was selected based on its cross-validation MAE of approximately ₹4.13 lakh. HistGradientBoosting achieved the highest test-set R² among the compared models.

*Note: These are dataset-specific evaluation results, not a guarantee of prediction accuracy for every vehicle.*

## 📁 Project Structure

```text
CarMetric/
│
├── app.py
│   └── Streamlit application for interactive car price prediction
│
├── train_model.py
│   └── Data preprocessing, model training and evaluation pipeline
│
├── Car Dataset Processed.csv
│   └── Processed dataset used for machine learning
│
├── final_model.joblib
│   └── Saved trained machine learning pipeline
│
├── model_evaluation.json
│   └── Model comparison and evaluation results
│
├── training_run_output.json
│   └── Training execution details and output
│
├── Untitled0.ipynb
│   └── Jupyter Notebook for exploratory analysis and experimentation
│
├── requirements.txt
│   └── Python project dependencies
│
├── .gitignore
│   └── Files and directories excluded from Git tracking
│
└── README.md
    └── Project documentation
```

## ⚙️ Installation and Setup

Follow these steps to run CarMetric locally.

### 1. Clone the Repository

```bash
git clone https://github.com/HimJaiss/CarMetric.git
```

### 2. Navigate to the Project Directory

```bash
cd CarMetric
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

## 🖥️ How to Use

1. Launch the CarMetric application.
2. Enter the required vehicle information in the input form.
3. Provide the relevant vehicle specifications and ownership details.
4. Click the **Estimate Resale Price** button.
5. View the predicted resale value displayed by the application.

## 📦 Training the Model

To retrain the machine learning models using the available dataset, execute:

```bash
python train_model.py
```

The training script performs preprocessing, model training, cross-validation, and evaluation. It also saves the trained model and evaluation outputs.

**Note:** Retraining may replace the existing model and evaluation files.

## 🔮 Future Improvements

- Expand the dataset with more diverse vehicle records.
- Perform deeper exploratory data analysis (EDA).
- Improve feature engineering and model optimization.
- Analyze residuals and prediction errors.
- Add feature importance and model explainability using SHAP.
- Experiment with additional regression algorithms.
- Improve the user interface with interactive visualizations.
- Deploy the application on a cloud platform.

## 🎯 Learning Outcomes

Through this project, I explored and implemented:

- End-to-end machine learning development.
- Data cleaning and preprocessing.
- Numerical and categorical feature handling.
- Regression algorithms and ensemble learning.
- Train-test splitting and cross-validation.
- Model evaluation using regression metrics.
- Saving and loading trained ML pipelines.
- Building interactive ML applications using Streamlit.

## 👨‍💻 Author

**Himanshu Jaiswal**

MCA Student | Aspiring Data Scientist | Software Developer

- GitHub: [@HimJaiss](https://github.com/HimJaiss)
- LinkedIn: [Connect with me](https://www.linkedin.com/)

---

<p align="center">
  <strong>⭐ If you find this project useful, consider giving it a star!</strong>
</p>

<p align="center">
  Built with ❤️ using Python and Machine Learning.
</p>