import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
df = pd.read_csv(url, names=column_names, na_values='?')
df.dropna(inplace=True)

# Encode categorical variables
df['sex'] = df['sex'].replace({1: 1, 0: 0})  # Convert 'sex' to numeric
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier().fit(X_train, y_train)

# Generate SHAP values for model explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_train, check_additivity=False)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the Dashboard
app.layout = dbc.Container([
    html.H1("Heart Disease Risk Prediction", className="text-center mt-3"),
    
    # Update button
    html.Button("Update Dashboard", id="update-button", n_clicks=0),
    
    # Header Section
    dbc.Row([
        dbc.Col(html.Div([
            html.H4("Dashboard Overview"),
            html.P("Summary of risk predictions and key patient metrics for heart disease."),
            dcc.Graph(id="risk-score-gauge")
        ]))
    ]),

    # Key Metrics Section
    dbc.Row([
        dbc.Col(dcc.Graph(id="hypertension-likelihood-chart"), md=6),
        dbc.Col(dcc.Graph(id="cv-health-indicator-chart"), md=6),
    ]),

    # Feature Exploration Panel
    dbc.Row([
        dbc.Col(dcc.Graph(id="feature-correlation-heatmap"), md=6),
        dbc.Col(dcc.Graph(id="exercise-tolerance-plot"), md=6),
    ]),

    # Patient Segmentation
    dbc.Row([
        dbc.Col(dcc.Graph(id="patient-clusters"), md=6),
        # dbc.Col(dcc.Graph(id="geographical-risk-map"), md=6),
    ]),

    # Actionable Insights
    dbc.Row([
        dbc.Col(html.Div([
            html.H5("Model Explanation"),
            dcc.Graph(id="shap-summary-plot")
        ]), md=6),
        dbc.Col(html.Div([
            html.H5("Recommendations for Clinicians"),
            html.Ul([
                html.Li("Consider additional testing for patients with high risk."),
                html.Li("Review lifestyle recommendations for medium-risk patients.")
            ])
        ]), md=6),
    ])
], fluid=True)

# Callbacks for Interactivity

# Risk Score Gauge
@app.callback(
    Output("risk-score-gauge", "figure"),
    Input("update-button", "n_clicks")
)
def update_risk_score_gauge(n_clicks):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0.6,  # Placeholder score
        title={"text": "Risk Score"},
        gauge={"axis": {"range": [0, 1]}
        }
    ))
    return fig

# Hypertension Likelihood Chart
@app.callback(
    Output("hypertension-likelihood-chart", "figure"),
    Input("update-button", "n_clicks")
)
def update_hypertension_chart(n_clicks):
    fig = px.bar(df, x="age", y="trestbps", title="Hypertension Likelihood by Age")
    return fig

# Cardiovascular Health Indicator Chart
@app.callback(
    Output("cv-health-indicator-chart", "figure"),
    Input("update-button", "n_clicks")
)
def update_cv_health_chart(n_clicks):
    fig = px.pie(df, values="chol", names="sex", title="Cardiovascular Health Indicator")
    return fig

# Feature Correlation Heatmap
@app.callback(
    Output("feature-correlation-heatmap", "figure"),
    Input("update-button", "n_clicks")
)
def update_correlation_heatmap(n_clicks):
    corr_matrix = pd.DataFrame(X_train).corr()
    fig = px.imshow(corr_matrix, title="Feature Correlation Heatmap")
    return fig

# Exercise Tolerance Plot
@app.callback(
    Output("exercise-tolerance-plot", "figure"),
    Input("update-button", "n_clicks")
)
def update_exercise_tolerance_plot(n_clicks):
    fig = px.scatter(df, x="thalach", y="oldpeak", title="Exercise Tolerance and Heart Disease Risk")
    return fig

# Patient Clusters
@app.callback(
    Output("patient-clusters", "figure"),
    Input("update-button", "n_clicks")
)
def update_patient_clusters(n_clicks):
    fig = px.scatter(df, x="age", y="chol", color=df['target'].astype(str),
                     title="Patient Clusters by Age and Cholesterol Level")
    fig.update_layout(coloraxis_colorbar=dict(title="Heart Disease (1=Yes, 0=No)"))
    return fig

# SHAP Summary Plot - converting SHAP plot to Plotly compatible format
@app.callback(
    Output("shap-summary-plot", "figure"),
    Input("update-button", "n_clicks")
)
def update_shap_summary(n_clicks):
    # Select SHAP values for class 1 (assuming binary classification with classes 0 and 1)
    shap_class_1_values = shap_values.values[..., 1]  # Selecting SHAP values for class 1
    
    # Convert SHAP values to a DataFrame
    shap_df = pd.DataFrame(shap_class_1_values, columns=[f'Feature {i}' for i in range(X_train.shape[1])])
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    
    # Create Plotly bar chart for feature importance based on SHAP values
    fig = px.bar(x=mean_abs_shap.index, y=mean_abs_shap.values,
                 title="Feature Importance based on SHAP Values (Class 1)")
    fig.update_layout(xaxis_title="Features", yaxis_title="Mean |SHAP Value|",
                      title="SHAP Summary - Feature Importance for Class 1")
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
