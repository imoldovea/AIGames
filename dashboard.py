import os

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

OUTPUT = "./output/"
INPUT = "./input/"
LOSS_FILE = os.path.join(OUTPUT, "loss_data.csv")

app = Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div(
    children=[
        html.H1(children="Real-Time Training Dashboard"),
        dcc.Graph(id="training-loss-graph"),
        dcc.Graph(id="validation-loss-graph"),
        dcc.Interval(
            id="interval-component",
            interval=5 * 1000,  # in milliseconds
            n_intervals=0,
        ),
    ]
)

@app.callback(
    Output("training-loss-graph", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_graph(n):
    """
    Updates the training loss graph with data from the loss file.
    Displays only the training loss.
    """
    try:
        df = pd.read_csv(LOSS_FILE)

        # Basic error handling (replace with more robust validation if needed)
        if df.empty:
            return {
                "data": [],
                "layout": {
                    "title": "No Training Data Available",
                    "xaxis": {"title": "Epoch"},
                    "yaxis": {"title": "Loss"},
                },
            }

        fig = go.Figure()
        for model in df["model"].unique():
            df_model = df[df["model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=df_model["epoch"],
                    y=df_model["loss"],  # Plot training loss
                    mode="lines+markers",
                    name=model,
                )
            )

        fig.update_layout(
            title="Real-Time Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
        )
        return fig
    except Exception as e:
        print(f"Error updating training loss graph: {e}")
        return {
            "data": [],
            "layout": {
                "title": f"Error Loading Training Data: {e}",
                "xaxis": {"title": "Epoch"},
                "yaxis": {"title": "Loss"},
            },
        }


@app.callback(
    Output("validation-loss-graph", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_validation_graph(n):
    """
    Updates the validation loss graph with data from the loss file.
    Displays only the validation loss.
    """
    try:
        df = pd.read_csv(LOSS_FILE)

        if df.empty:
            return {
                "data": [],
                "layout": {
                    "title": "No Validation Data Available",
                    "xaxis": {"title": "Epoch"},
                    "yaxis": {"title": "Validation Loss"},
                },
            }

        fig = go.Figure()
        for model in df["model"].unique():
            df_model = df[df["model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=df_model["epoch"],
                    y=df_model["validation_loss"],  # Plot validation loss
                    mode="lines+markers",
                    name=model,
                )
            )

        fig.update_layout(
            title="Real-Time Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Validation Loss",
        )
        return fig
    except Exception as e:
        print(f"Error updating validation loss graph: {e}")
        return {
            "data": [],
            "layout": {
                "title": f"Error Loading Validation Data: {e}",
                "xaxis": {"title": "Epoch"},
                "yaxis": {"title": "Validation Loss"},
            },
        }


def validate_loss_file(file_path: str) -> bool:
    """
    Validates that the loss file exists and contains the necessary columns.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Loss file not found: {file_path}")
            return False

        df = pd.read_csv(file_path)
        required_columns = {"model", "epoch", "loss", "validation_loss"}  # Corrected column check
        if not required_columns.issubset(df.columns):
            print(f"Loss file missing required columns: {required_columns}")
            return False

        return True

    except Exception as e:
        print(f"Error validating loss file: {e}")
        return False


if __name__ == "__main__":
    if validate_loss_file(LOSS_FILE):
        app.run(debug=True)
    else:
        print("Loss file validation failed.  Please check the file and its contents.")
