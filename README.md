# 🏠 House Price Prediction using Machine Learning

A Machine Learning project that predicts house prices based on property features using multiple regression models. The project includes data preprocessing, exploratory data analysis (EDA), model comparison, visualization, and a production-ready prediction pipeline built with Scikit-learn.

---

## 📌 Project Overview

This project predicts the selling price of houses using historical housing data. Multiple machine learning algorithms were trained and evaluated, and the best-performing model was automatically selected and saved for future predictions.

The project follows an end-to-end machine learning workflow, including:

- Data Loading
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation
- Model Comparison
- Model Saving
- House Price Prediction using a Saved Pipeline

---

## 📂 Dataset

Dataset: **Housing.csv**

Features include:

- Bedrooms
- Bathrooms
- Living Area
- Lot Size
- Floors
- Waterfront
- View
- Condition
- Above Ground Area
- Basement Area
- Year Built
- Year Renovated
- City
- State ZIP
- Date

Target Variable:

- **Price**

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 🤖 Machine Learning Models

The following regression models were trained and compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best model is selected automatically based on the highest **R² Score**.

---

## ⚙️ Data Preprocessing

The project uses a production-ready Scikit-learn Pipeline.

### Preprocessing includes:

- Date Feature Extraction (Year, Month, Day)
- Removal of High Cardinality Features (Street)
- Removal of Constant Features (Country)
- One-Hot Encoding using `OneHotEncoder(handle_unknown="ignore")`
- Standard Scaling for numerical features where applicable

---

## 📊 Exploratory Data Analysis (EDA)

The following visualizations are generated automatically:

- Price Distribution
- Correlation Heatmap
- Pair Plot
- Actual vs Predicted Plot
- Residual Plot
- Feature Importance Plot

All plots are stored inside the **plots/** folder.

---

## 📈 Model Evaluation Metrics

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The project automatically compares all models and selects the best one.

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── archive/
│   └── Housing.csv
│
├── plots/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── pairplot.png
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── feature_importance.png
│
├── main.py
├── inference.py
├── pipeline_components.py
├── best_model.pkl
├── house_price_prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

### Clone Repository

```bash
git clone <repository-url>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python main.py
```

### Predict House Price

```bash
python inference.py
```

---

## 📌 Sample Output

```
Linear Regression

MAE : ...
RMSE : ...
R² Score : ...

Random Forest

MAE : ...
RMSE : ...
R² Score : ...

Gradient Boosting

MAE : ...
RMSE : ...
R² Score : ...

Best Model:
Gradient Boosting

Predicted House Price:
₹402,091.06
```

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- XGBoost Regressor
- LightGBM
- Cross Validation
- Model Deployment using Flask/FastAPI
- Interactive Web Application

---

## 👨‍💻 Author

**Nikhil**

Final Year Computer Science Engineering Student

University of Lucknow

---

## ⭐ If you found this project useful, don't forget to star the repository.
