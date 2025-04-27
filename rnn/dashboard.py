# dashboard.py
import logging
import os
from configparser import ConfigParser

import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html, no_update
from dash.dependencies import Input, Output

# File path to the CSV file
PARAMETERS_FILE = "config.properties"
config = ConfigParser()

config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
LOSS_DATA = config.get("FILES", "LOSS_DATA", fallback="loss_data.csv")
LOSS_FILE = f"{OUTPUT}{LOSS_DATA}"
LOSS_CHART = f"{OUTPUT}loss_chart.html"

def load_loss_data():
    try:
        if not os.path.exists(LOSS_FILE):
            raise FileNotFoundError(f"Loss file not found: {LOSS_FILE}")
        df = pd.read_csv(LOSS_FILE)

        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df['train_loss'] = pd.to_numeric(df['train_loss'], errors='coerce')
        df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')
        df['train_acc'] = pd.to_numeric(df['train_acc'], errors='coerce')
        df['val_acc'] = pd.to_numeric(df['val_acc'], errors='coerce')
        df['time_per_step'] = (pd.to_numeric(df['time_per_step'], errors='coerce') / 60).round(0)
    except Exception as e:
        df = pd.DataFrame()
    return df


# Initialize the Dash app
app = dash.Dash(__name__)

# Dashboard layout with five graphs (added time_per_step)
app.layout = html.Div([
    html.H1("Neural Network Training Dashboard"),
    dcc.Graph(id="training-loss-graph"),
    dcc.Graph(id="validation-loss-graph"),
    dcc.Graph(id="accuracy-graph"),
    dcc.Graph(id="validation-accuracy-graph"),
    dcc.Graph(id="time-per-step-graph"),
    dcc.Graph(id="exit-weight-graph"),  # New graph
    dcc.Interval(id="interval-component", interval=3000, n_intervals=0)
])


# Update the callback to include exit weight
@app.callback(
    Output("training-loss-graph", "figure"),
    Output("validation-loss-graph", "figure"),
    Output("accuracy-graph", "figure"),
    Output("validation-accuracy-graph", "figure"),
    Output("time-per-step-graph", "figure"),
    Output("exit-weight-graph", "figure"),  # New output
    Input("interval-component", "n_intervals")
)


def update_graphs(n):
    df = load_loss_data()
    if df.empty or 'model_name' not in df.columns:
        logging.debug("No training data available yet. Waiting for loss_data.csv...")
        return no_update, no_update, no_update, no_update, no_update

    df = df.sort_values('epoch')

    # Each figure now includes the 'model' column as the color dimension
    fig_training = px.line(
        df,
        x='epoch',
        y='train_loss',
        color='model_name',
        line_dash='model_name',
        symbol='model_name',
        markers=True,
        title="Training Loss"
    )
    fig_validation = px.line(
        df,
        x='epoch',
        y='val_loss',
        color='model_name',
        line_dash='model_name',
        symbol='model_name',
        markers=True,
        title="Validation Loss"
    )
    fig_accuracy = px.line(
        df,
        x='epoch',
        y='train_acc',
        color='model_name',
        line_dash='model_name',
        symbol='model_name',
        markers=True,
        title="Training Accuracy %"
    )
    fig_val_accuracy = px.line(
        df,
        x='epoch',
        y='val_acc',
        color='model_name',
        line_dash='model_name',
        symbol='model_name',
        markers=True,
        title="Validation Accuracy %"
    )
    fig_time_per_step = px.line(
        df,
        x='epoch',
        y='time_per_step',
        color='model_name',
        markers=True,
        title="Time Per Step (minutes)"
    )
    fig_time_per_step.update_yaxes(title_text="time_per_step (minutes)")

    # Update layout for each figure:
    for fig in (fig_training, fig_validation, fig_accuracy, fig_val_accuracy, fig_time_per_step):
        fig.update_layout(
            xaxis_title="Epoch",
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                traceorder="normal"
            )
        )

    # Add new exit weight figure
    fig_exit_weight = px.line(
        df,
        x='epoch',
        y='exit_weight',
        color='model_name',
        line_dash='model_name',
        symbol='model_name',
        markers=True,
        title="Exit Weight per Epoch"
    )
    fig_exit_weight.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Exit Weight",
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            traceorder="normal")
    )

    return fig_training, fig_validation, fig_accuracy, fig_val_accuracy, fig_time_per_step, fig_exit_weight


if __name__ == '__main__':
    app.run(debug=True)  # Changed from app.run_server(debug=True)
