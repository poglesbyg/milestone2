# Heart Disease Prediction: A Research Plan

## 1. Introduction

Cardiovascular disease remains a principal cause of mortality on a global scale, necessitating advancements in early diagnostic capabilities to improve patient outcomes. This research aims to predict the likelihood of heart disease by analyzing patient demographic and medical attributes through advanced machine learning methodologies. By harnessing data analytics, we intend to elucidate critical patterns and provide actionable insights that may enhance early detection and intervention strategies for cardiovascular conditions.

## 2. Research Question

**To what extent can we accurately predict the risk of heart disease in patients by analyzing demographic and clinical variables from the UCI Heart Disease dataset?**

The objective of this research is to identify significant features that contribute to heart disease risk and evaluate predictive models to determine their efficacy. Insights gained from this analysis could facilitate healthcare professionals in identifying high-risk individuals and optimizing personalized preventive strategies.

## 3. Virtual Environment Setup

To ensure reproducibility and appropriate management of dependencies, a virtual environment is employed using Python's `venv` module.

```bash
# Create virtual environment
python3 -m venv heart_env

# Activate virtual environment
source heart_env/bin/activate  # On Windows use: heart_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 4. Data Source

The dataset utilized in this project is the **UCI Heart Disease dataset**, which is accessible at [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). This dataset contains several clinical and demographic attributes that are hypothesized to have predictive power for heart disease risk.

The following script demonstrates how to load and preprocess the dataset:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset using pandas
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load dataset into a DataFrame
df = pd.read_csv(url, names=column_names, na_values='?')

# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
df['sex'] = df['sex'].replace({1: 'male', 0: 'female'})
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Dataset Attributes
The dataset comprises 303 patient records, each characterized by 14 attributes:

- **Age**: Age in years
- **Sex**: Gender (1 = male, 0 = female)
- **Chest Pain Type (cp)**: Types of chest pain (e.g., typical angina, asymptomatic)
- **Resting Blood Pressure (trestbps)**: Resting blood pressure in mm Hg
- **Cholesterol (chol)**: Serum cholesterol level in mg/dl
- **Fasting Blood Sugar (fbs)**: Indicator if fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **Resting ECG (restecg)**: Results from resting electrocardiographic tests
- **Max Heart Rate Achieved (thalach)**: Maximum heart rate achieved during exercise
- **Exercise Induced Angina (exang)**: Presence of exercise-induced angina (1 = yes, 0 = no)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope of Peak Exercise ST Segment (slope)**: Characterizes the slope (e.g., upsloping, flat, downsloping)
- **Number of Major Vessels (ca)**: Number of major vessels colored by fluoroscopy (0-3)
- **Thalassemia (thal)**: Normal, fixed defect, or reversible defect

The dependent variable, **'target'**, indicates the presence or absence of heart disease, allowing for a binary classification problem.

## 5. Data Preparation

Data preparation involves transforming the dataset to facilitate analysis, including managing missing values, encoding categorical variables, and deriving additional features that may provide further insight into patient health.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset into a Pandas DataFrame
heart_disease = fetch_ucirepo(id=45)

# Ensure data is not None
df = heart_disease.data
if df is None or not isinstance(df, pd.DataFrame):
    raise ValueError("Failed to load the dataset or the dataset is not in a valid format. Please check the dataset source and try again.")

# Handle missing values and encode categorical variables
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].replace({1: 'male', 0: 'female'})
df = pd.get_dummies(df, columns=['cp'], drop_first=True)

# Adding derived outcomes to the dataset

# 1. Risk Assessment for Cardiovascular Diseases
w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.2, 0.2
df['risk_score'] = w1 * df['age'] + w2 * df['chol'] + w3 * df['trestbps'] + w4 * df['fbs'] + w5 * df['thalach']

# 2. Likelihood of Hypertension
df['hypertension_likelihood'] = ((df['trestbps'] > 130) | (df['chol'] > 200)) & (df['age'] > 40)

# 3. Metabolic Syndrome Risk
df['metabolic_syndrome_risk'] = (df['chol'] > 200).astype(int) + (df['fbs'] > 120).astype(int) + (df['trestbps'] > 130).astype(int)

# 4. Exercise Tolerance and Cardiovascular Fitness
df['exercise_tolerance_score'] = df['thalach'] - (df['age'] * 0.5) - (df['exang'] * 20)

# 5. Stress or Anxiety Indicator
df['stress_indicator'] = ((df['cp'] == 2) | (df['cp'] == 3)) & (df['restecg'] == 1)

# 6. Predicting Coronary Artery Disease (CAD) Severity
df['cad_severity_score'] = 3 * df['thal'] + 2 * df['oldpeak'] + df['exang'] + df['restecg']

# 7. Heart Function and Electrical Stability
df['heart_instability_indicator'] = (df['restecg'] == 1).astype(int)

# 8. Prediction of Diabetes or Pre-Diabetic State
df['diabetes_risk'] = (df['fbs'] > 120).astype(int)

# 9. Atherosclerosis Likelihood
df['atherosclerosis_score'] = ((df['age'] > 45).astype(int) + (df['chol'] > 240).astype(int) + (df['trestbps'] > 140).astype(int))

# 10. Exercise-Induced Ischemia
df['ischemia_indicator'] = ((df['oldpeak'] > 1.0) & (df['exang'] == 1)).astype(int)

# 11. General Cardiovascular Health Indicator
w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
df['health_score'] = 100 - (w1 * df['age'] + w2 * df['trestbps'] + w3 * df['chol'] + w4 * df['fbs'])

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) provides a fundamental understanding of the dataset, helps identify patterns, and elucidates relationships between variables, which are pivotal for model development and feature selection.

### Feature Distributions

We perform visualization of feature distributions to assess variability, detect anomalies, and identify potential outliers that could influence model performance.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of key features
plt.figure(figsize=(15, 5))
sns.histplot(df['age'], kde=True, bins=30, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(15, 5))
sns.histplot(df['chol'], kde=True, bins=30, color='green')
plt.title('Cholesterol Distribution')
plt.xlabel('Cholesterol (mg/dl)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(15, 5))
sns.histplot(df['trestbps'], kde=True, bins=30, color='red')
plt.title('Resting Blood Pressure Distribution')
plt.xlabel('Blood Pressure (mm Hg)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(15, 5))
sns.histplot(df['thalach'], kde=True, bins=30, color='purple')
plt.title('Maximum Heart Rate Distribution')
plt.xlabel('Max Heart Rate')
plt.ylabel('Frequency')
plt.show()

# Explore correlations between features
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Detect and handle outliers using boxplots
plt.figure(figsize=(15, 5))
sns.boxplot(x=df['chol'])
plt.title('Cholesterol Boxplot for Outlier Detection')
plt.show()

plt.figure(figsize=(15, 5))
sns.boxplot(x=df['trestbps'])
plt.title('Resting Blood Pressure Boxplot for Outlier Detection')
plt.show()
```

### Insights from EDA

Through EDA, we identified significant distributions and correlations among features, which are critical for feature engineering and model development. Observing these relationships provides a basis for selecting the most important predictors of heart disease and informs model selection strategies.

## 7. Summary

This research seeks to predict the likelihood of heart disease using patient demographic and clinical data, utilizing machine learning techniques to achieve this objective. By systematically preparing data, conducting exploratory analysis, and developing predictive models, this

