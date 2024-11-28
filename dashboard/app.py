# app.py
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
from gauge import create_gauge  # Import the gauge function

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample risk score (you would replace this with model calculations)
default_risk_score = 0.6

# Layout of the Dashboard
app.layout = dbc.Container([
    html.H1("Heart Disease Risk Prediction", className="text-center mt-3"),
    
    # Patient Input Section
    dbc.Row([
        dbc.Col(html.Div([
            html.H5("Patient Information"),
            html.Label("Age:"),
            dcc.Input(id="input-age", type="number", placeholder="Enter age", min=0, max=120, step=1),
            html.Label("Sex:"),
            dcc.Dropdown(id="input-sex", options=[
                {'label': 'Male', 'value': 1},
                {'label': 'Female', 'value': 0}
            ], placeholder="Select sex"),
            html.Label("Weight (kg):"),
            dcc.Input(id="input-weight", type="number", placeholder="Enter weight", min=0, max=200, step=1),
            html.Button("Calculate Patient Risk", id="calculate-button", n_clicks=0, className="mt-2")
        ]), width=4),
        dbc.Col(html.Div([
            dcc.Graph(id="risk-score-gauge", figure=create_gauge(default_risk_score), style={"height": "500px", "width": "100%"}),
            html.P(
                "The Risk Score Gauge represents the predicted probability of heart disease "
                "based on the input patient details or the average test set probability. A higher score indicates a greater likelihood "
                "of heart disease. The gauge is color-coded: green (low risk), yellow (moderate risk), "
                "and red (high risk), helping clinicians quickly assess overall heart disease risk."
            )
        ]), width=8),
    ])
], fluid=True)

# Callback to update gauge based on patient input
@app.callback(
    Output("risk-score-gauge", "figure"),
    Input("calculate-button", "n_clicks"),
    [State("input-age", "value"), State("input-sex", "value"), State("input-weight", "value")]
)
def update_risk_score_gauge(n_clicks, age, sex, weight):
    if n_clicks > 0 and age is not None and sex is not None and weight is not None:
        # Sample calculation for patient-specific risk score (replace with actual model calculation)
        risk_score = 0.7  # Example, you would calculate this based on patient data
    else:
        # Default average risk score
        risk_score = default_risk_score

    return create_gauge(risk_score)

if __name__ == "__main__":
    app.run_server(debug=True)
