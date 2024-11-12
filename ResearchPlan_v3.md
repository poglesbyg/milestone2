# Heart Disease Prediction: Research Plan

## 1. Introduction

Heart disease is one of the leading causes of mortality worldwide, and early detection plays a crucial role in improving patient outcomes. By analyzing medical data, we can identify patterns and risk factors that contribute to heart disease. This project aims to explore whether we can predict the likelihood of heart disease based on various demographic and medical attributes using data from the UCI Heart Disease dataset. The goal is to leverage data analytics and machine learning to contribute to the understanding and early detection of heart disease.

## 2. Research Question

**Can we predict the likelihood of heart disease based on patient demographic and medical data using the UCI Heart Disease dataset?**

This research question will help identify the most relevant features that contribute to heart disease and explore how accurately we can predict its occurrence. The insights gained from this analysis could assist healthcare professionals in assessing heart disease risk in patients and making more informed decisions.

## 3. Virtual Environment Setup

To ensure that all dependencies are managed properly and to maintain a clean Python environment, we will create a virtual environment using `venv`.

```bash
# Create virtual environment
python3 -m venv heart_env

# Activate virtual environment
source heart_env/bin/activate  # On Windows use: heart_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 4. Data Source

For this analysis, we will use the **Heart Disease dataset** from the UCI Machine Learning Repository, accessible at [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). The dataset can be fetched using the following Python code:

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
# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
df['sex'] = df['sex'].replace({1: 'male', 0: 'female'})
if 'cp' in df.columns:
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

The dataset contains 303 observations, each representing a patient, with 14 attributes, including:

- **Age**: Age in years
- **Sex**: Gender (1 = male, 0 = female)
- **Chest Pain Type (cp)**: Four types of chest pain (e.g., typical angina, asymptomatic)
- **Resting Blood Pressure (trestbps)**: Measured in mm Hg
- **Cholesterol (chol)**: Serum cholesterol level in mg/dl
- **Fasting Blood Sugar (fbs)**: Whether fasting blood sugar is > 120 mg/dl
- **Resting ECG (restecg)**: Resting electrocardiographic results
- **Max Heart Rate Achieved (thalach)**: Maximum heart rate during exercise
- **Exercise Induced Angina (exang)**: Whether angina occurred during exercise
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope of Peak Exercise ST Segment (slope)**: Upsloping, flat, or downsloping
- **Number of Major Vessels (ca)**: Number of major vessels colored by fluoroscopy (0-3)
- **Thalassemia (thal)**: Normal, fixed defect, or reversible defect

The target variable in the dataset is named 'target', which indicates the presence or absence of heart disease. This allows us to develop a classification model to predict this condition.

## 5. Data Preparation

In this section, we prepare the data for analysis by handling missing values, encoding categorical variables, and adding derived outcomes to enrich the dataset.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset into a Pandas DataFrame
heart_disease = fetch_ucirepo(id=45)

# Ensure data is not None
df = heart_disease.data
if df is None or not isinstance(df, pd.DataFrame):
    raise ValueError("Failed to load the dataset or the dataset is not in a valid format. Please check the dataset source and try again.")

# Handle missing values and encode categorical variables (if they exist in the dataset)
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].replace({1: 'male', 0: 'female'})
if 'cp' in df.columns:
    df = pd.get_dummies(df, columns=['cp'], drop_first=True)

# Adding derived outcomes as new columns to the dataset

# 1. Risk Assessment for Cardiovascular Diseases
w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.2, 0.2
# Assuming weights are equal for simplicity
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
w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25  # Assuming equal weights for simplicity
df['health_score'] = 100 - (w1 * df['age'] + w2 * df['trestbps'] + w3 * df['chol'] + w4 * df['fbs'])

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis helps in understanding the structure and characteristics of the data. It also aids in uncovering patterns and relationships among features that could be valuable for predictive modeling.

### Feature Distributions

We visualize the distributions of key features to understand their spread and identify any anomalies or outliers that might affect the model's performance.

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

plt.figure

```