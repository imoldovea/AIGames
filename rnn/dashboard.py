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

def load_loss_data():
    try:
        df = pd.read_csv(LOSS_FILE)
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df['training_loss'] = pd.to_numeric(df['training_loss'], errors='coerce')
        df['validation_loss'] = pd.to_numeric(df['validation_loss'], errors='coerce')
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        df['validation_accuracy'] = pd.to_numeric(df['validation_accuracy'], errors='coerce')
        df['time_per_step'] = (pd.to_numeric(df['time_per_step'], errors='coerce') / 60).round(0)
    except Exception as e:
        print("Error loading CSV:", e)
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
    dcc.Interval(id="interval-component", interval=3000, n_intervals=0)
])


@app.callback(
    Output("training-loss-graph", "figure"),
    Output("validation-loss-graph", "figure"),
    Output("accuracy-graph", "figure"),
    Output("validation-accuracy-graph", "figure"),
    Output("time-per-step-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_graphs(n):
    df = load_loss_data()
    if df.empty or 'model' not in df.columns:
        return no_update, no_update, no_update, no_update, no_update

    df = df.sort_values('epoch')

    # Each figure now includes the 'model' column as the color dimension
    fig_training = px.line(
        df,
        x='epoch',
        y='training_loss',
        color='model',
        markers=True,
        title="Training Loss"
    )
    fig_validation = px.line(
        df,
        x='epoch',
        y='validation_loss',
        color='model',
        markers=True,
        title="Validation Loss"
    )
    fig_accuracy = px.line(
        df,
        x='epoch',
        y='accuracy',
        color='model',
        markers=True,
        title="Accuracy"
    )
    fig_val_accuracy = px.line(
        df,
        x='epoch',
        y='validation_accuracy',
        color='model',
        markers=True,
        title="Validation Accuracy"
    )
    fig_time_per_step = px.line(
        df,
        x='epoch',
        y='time_per_step',
        color='model',
        markers=True,
        title="Time Per Step"
    )
    fig_time_per_step.update_yaxes(title_text="time_per_step (minutes)")

    # Update layout for each figure:
    for fig in (fig_training, fig_validation, fig_accuracy, fig_val_accuracy, fig_time_per_step):
        fig.update_layout(
            xaxis_title="Epoch",
            legend=dict(
                x=1.02,  # Place the legend to the right of the plot
                y=1,
                traceorder="normal"
            )
        )

    return fig_training, fig_validation, fig_accuracy, fig_val_accuracy, fig_time_per_step


if __name__ == '__main__':
    app.run_server(debug=True)
