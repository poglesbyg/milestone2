# gauge.py
import plotly.graph_objects as go

def create_gauge(risk_score):
    # Create the gauge chart with additional features
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        title={"text": "Heart Disease Risk Score", "font": {"size": 24}},
        delta={"reference": 0.5, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "darkred"},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 0.3], "color": "lightgreen", "name": "Low Risk"},
                {"range": [0.3, 0.6], "color": "yellow", "name": "Moderate Risk"},
                {"range": [0.6, 1], "color": "red", "name": "High Risk"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": risk_score
            }
        }
    ))

    # Update layout for a larger display
    fig.update_layout(
        font={'size': 16},
        height=500,  # Set height of the gauge chart
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig
