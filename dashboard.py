import logging

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import numpy as np  # Import NumPy for NaN handling
import os  # Import os module for file operations
import traceback
import plotly.io as pio



# File where training script logs loss values
OUTPUT = "output/"
INPUT = "input/"
LOSS_FILE = f"{OUTPUT}loss_data.csv"

# Initialize Dash app
app = dash.Dash(__name__)
app.logger.setLevel(logging.ERROR)
app.logger.disabled = True


# Layout of the web application
app.layout = html.Div([
    html.H1("Real-Time Training Loss Visualization"),
    dcc.Graph(id="loss-graph"),
    dcc.Graph(id="validation-loss-graph"),  # Added second graph for validation_loss
    dcc.Interval(
        id="interval-component",
        interval=10000,  # Update every 10 seconds
        n_intervals=0
    )
])

@app.callback(
    Output("loss-graph", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_graph(n_intervals):
    try:
        # Read the loss data from CSV
        df = pd.read_csv(LOSS_FILE)

        # Verify that the required columns are present
        required_columns = {'model', 'epoch', 'loss', 'validation_loss', 'time'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")

        df["time_minutes"] = df["time"] / 6e7

        # Initialize the graph
        fig = go.Figure()

        # For each unique model in the data, add traces for training and validation loss.
        models = df['model'].unique()
        for model in models:
            model_df = df[df['model'] == model]
            training_time = df["time_minutes"] = df["time"] / 6e7
            hover_text = [f"Epoch time: {t:.2f} min" for t in model_df["time_minutes"]]

            # Training loss trace
            fig.add_trace(
                go.Scatter(
                    x=model_df["epoch"],
                    y=model_df["loss"],
                    mode="lines+markers",
                    name=f"{model} - Loss",
                    text=hover_text,
                    hovertemplate="Epoch: %{x}<br>Loss: %{y}<br>%{text}<extra></extra>"
                )
            )

            # Validation loss trace
            fig.add_trace(
                go.Scatter(
                    x=model_df["epoch"],
                    y=model_df["validation_loss"],
                    mode="lines+markers",
                    name=f"{model} - Validation Loss",
                    text=hover_text,
                    hovertemplate="Epoch: %{x}<br>Validation Loss: %{y}<br>%{text}<extra></extra>"
                )
            )

        # Update the layout for better visualization
        fig.update_layout(
            title="Training and Validation Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template=pio.templates.default,
            legend_title="Models"
        )

        # Return the updated figure so it can be rendered
        return fig

    except Exception as e:
        logging.error("Error updating graph:")
        logging.error(traceback.format_exc())
        return None


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
        return go.Figure()


def validate_loss_file():
    """
    Validates the existence, structure, and content of the loss data file.
    Terminates the process if the file does not exist, has incorrect columns, or is empty.
    """
    try:
        # Check if the loss file exists
        if not os.path.exists(LOSS_FILE):
            print(f"Error: File '{LOSS_FILE}' does not exist.")
            exit(1)

        # Read the file into a DataFrame
        df = pd.read_csv(LOSS_FILE)

        # Check for required columns
        required_columns = {'model', 'epoch', 'loss', 'validation_loss', 'time'}
        if not required_columns.issubset(df.columns):
            logging.warning(f"Error: File '{LOSS_FILE}' is missing required columns. "
                  f"Expected columns: {required_columns}.")
            exit(1)

        # Check if the file has at least one record
        if df.empty:
            logging.warning(f"Warning: File '{LOSS_FILE}' is currently empty. Waiting for new training data...")
            return  # Allow the process to continue without exiting


    except Exception as e:
        logging.warning(f"Error validating the loss file: {e}")
        exit(1)


# Run the Dash app
if __name__ == "__main__":
    validate_loss_file()  # Validate the loss file before starting the app
    app.run_server(debug=False, dev_tools_ui=False, dev_tools_hot_reload=False)
