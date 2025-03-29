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
    [
        html.H1("Training Dashboard"),

        # Existing components
        html.Div([
            html.H2("Training Loss"),
            dcc.Graph(id="training-loss-graph"),
        ]),

        html.Div([
            html.H2("Validation Loss"),
            dcc.Graph(id="validation-loss-graph"),
        ]),

        # New components for accuracy
        html.Div([
            html.H2("Training Accuracy"),
            dcc.Graph(id="training-accuracy-graph"),
        ]),

        html.Div([
            html.H2("Validation Accuracy"),
            dcc.Graph(id="validation-accuracy-graph"),
        ]),

        # Interval component
        dcc.Interval(
            id="interval-component",
            interval=5000,  # in milliseconds
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


@app.callback(
    Output("training-accuracy-graph", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_accuracy_graph(n):
    """
    Updates the training accuracy graph with data from the loss file.
    Displays only the training accuracy.
    """
    try:
        df = pd.read_csv(LOSS_FILE)

        # Basic error handling
        if df.empty or "accuracy" not in df.columns:
            return {
                "data": [],
                "layout": {
                    "title": "No Accuracy Data Available",
                    "xaxis": {"title": "Epoch"},
                    "yaxis": {"title": "Accuracy"},
                },
            }

        fig = go.Figure()
        for model in df["model"].unique():
            df_model = df[df["model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=df_model["epoch"],
                    y=df_model["accuracy"],
                    mode="lines+markers",
                    name=model,
                )
            )

        fig.update_layout(
            title="Real-Time Training Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
        )
        return fig
    except Exception as e:
        print(f"Error updating training accuracy graph: {e}")
        return {
            "data": [],
            "layout": {
                "title": f"Error Loading Accuracy Data: {e}",
                "xaxis": {"title": "Epoch"},
                "yaxis": {"title": "Accuracy"},
            },
        }


@app.callback(
    Output("validation-accuracy-graph", "figure"),
    Input("interval-component", "n_intervals"),
)
def update_validation_accuracy_graph(n):
    """
    Updates the validation accuracy graph with data from the loss file.
    Displays only the validation accuracy.
    """
    try:
        df = pd.read_csv(LOSS_FILE)

        if df.empty or "validation_accuracy" not in df.columns:
            return {
                "data": [],
                "layout": {
                    "title": "No Validation Accuracy Data Available",
                    "xaxis": {"title": "Epoch"},
                    "yaxis": {"title": "Validation Accuracy"},
                },
            }

        fig = go.Figure()
        for model in df["model"].unique():
            df_model = df[df["model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=df_model["epoch"],
                    y=df_model["validation_accuracy"],
                    mode="lines+markers",
                    name=model,
                )
            )

        fig.update_layout(
            title="Real-Time Validation Accuracy",
            xaxis_title="Epoch",
            yaxis_title="Validation Accuracy",
        )
        return fig
    except Exception as e:
        print(f"Error updating validation accuracy graph: {e}")
        return {
            "data": [],
            "layout": {
                "title": f"Error Loading Validation Accuracy Data: {e}",
                "xaxis": {"title": "Epoch"},
                "yaxis": {"title": "Validation Accuracy"},
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
        required_columns = {"model", "epoch", "loss", "validation_loss", "time", "accuracy",
                            "validation_accuracy"}  # Corrected column check
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
