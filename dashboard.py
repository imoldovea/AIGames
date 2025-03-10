import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import numpy as np  # Import NumPy for NaN handling

# File where training script logs loss values
OUTPUT = "output/"
INPUT = "input/"
LOSS_FILE = f"{OUTPUT}loss_data.csv"

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the web application
app.layout = html.Div([
    html.H1("Real-Time Training Loss Visualization"),
    dcc.Graph(id="loss-graph"),
    dcc.Graph(id="validation-loss-graph"),  # Added second graph for validation_loss
    dcc.Interval(
        id="interval-component",
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    )
])

# Callback function to update the graph
@app.callback(
    Output("loss-graph", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_graph(n_intervals):
    try:
        # Read the loss file
        df = pd.read_csv(LOSS_FILE)

        # Get unique models
        models = df['model'].unique()

        fig = go.Figure()

        # Add a trace for each model with breaks
        for model in models:
            model_df = df[df['model'] == model].copy()

            # Ensure gaps between independent runs
            model_df["epoch_diff"] = model_df["epoch"].diff()
            model_df.loc[model_df["epoch_diff"] > 1, "loss"] = np.nan  # Introduce NaN for breaks

            fig.add_trace(go.Scatter(
                x=model_df["epoch"],
                y=model_df["loss"],
                mode="lines+markers",
                name=model  # Use the model name for the legend
            ))

        fig.update_layout(
            title="Training Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark",
            legend_title="Models"  # Add a title to the legend
        )

        return fig

    except Exception as e:
        print(f"Error reading the loss data: {e}")
        return go.Figure()


# Callback function to update the validation loss graph
@app.callback(
    Output("validation-loss-graph", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_validation_graph(n_intervals):
    try:
        # Read the loss file
        df = pd.read_csv(LOSS_FILE)

        # Get unique models
        models = df['model'].unique()

        fig = go.Figure()

        # Add a trace for each model with breaks
        for model in models:
            model_df = df[df['model'] == model].copy()

            # Ensure gaps between independent runs
            model_df["epoch_diff"] = model_df["epoch"].diff()
            model_df.loc[model_df["epoch_diff"] > 1, "validation_loss"] = np.nan  # Introduce NaN for breaks

            fig.add_trace(go.Scatter(
                x=model_df["epoch"],
                y=model_df["validation_loss"],
                mode="lines+markers",
                name=model  # Use the model name for the legend
            ))

        fig.update_layout(
            title="Validation Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Validation Loss",
            template="plotly_dark",
            legend_title="Models"  # Add a title to the legend
        )

        return fig

    except Exception as e:
        print(f"Error reading the validation loss data: {e}")


# Run the Dash app
if __name__ == "__main__":
    app.run_server(debug=False)
