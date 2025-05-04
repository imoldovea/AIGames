# dashboard.py
import logging
import os
from configparser import ConfigParser
from io import BytesIO

import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State

import utils

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
            # logging.error(f"File not found: {LOSS_FILE}")
            raise FileNotFoundError(f"Loss file not found: {LOSS_FILE}")
        df = pd.read_csv(LOSS_FILE)

        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        df['train_loss'] = pd.to_numeric(df['train_loss'], errors='coerce')
        df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')
        df['train_acc'] = pd.to_numeric(df['train_acc'], errors='coerce')
        df['val_acc'] = pd.to_numeric(df['val_acc'], errors='coerce')
        df['time_per_step'] = (pd.to_numeric(df['time_per_step'], errors='coerce') / 60).round(0)
        df['collision_penalty'] = pd.to_numeric(df['collision_penalty'], errors='coerce')
        df['collision_rate'] = pd.to_numeric(df['collision_rate'], errors='coerce')
    except Exception as e:
        # logging.error(f"Error accessing loss data file: {e}")
        df = pd.DataFrame()

    # logging.info(f"Data preview:\n{df.head()}, Ecpoch: {len(df)}")  # Log first five rows
    return df


# Initialize the Dash app
app = dash.Dash(__name__)

# Modify your app layout to include download buttons and a Download component
app.layout = html.Div([
    html.H1("Neural Network Training Dashboard"),

    # First graph with download button
    html.Div([
        dcc.Graph(id="training-loss-graph"),
        html.Button("Download Training Loss", id="btn-download-training-loss", className="download-btn"),
    ]),

    # Second graph with download button
    html.Div([
        dcc.Graph(id="validation-loss-graph"),
        html.Button("Download Validation Loss", id="btn-download-validation-loss", className="download-btn"),
    ]),

    # Third graph with download button
    html.Div([
        dcc.Graph(id="accuracy-graph"),
        html.Button("Download Accuracy", id="btn-download-accuracy", className="download-btn"),
    ]),

    # Fourth graph with download button
    html.Div([
        dcc.Graph(id="validation-accuracy-graph"),
        html.Button("Download Validation Accuracy", id="btn-download-validation-accuracy", className="download-btn"),
    ]),

    # Fifth graph with download button
    html.Div([
        dcc.Graph(id="time-per-step-graph"),
        html.Button("Download Time per Step", id="btn-download-time-per-step", className="download-btn"),
    ]),

    # Sixth graph with download button
    html.Div([
        dcc.Graph(id="exit-weight-graph"),
        html.Button("Download Exit Weight", id="btn-download-exit-weight", className="download-btn"),
    ]),

    # Seventh graph with download button
    html.Div([
        dcc.Graph(id="collision-penalty-graph"),
        html.Button("Download Collision Penalty", id="btn-download-collision-penalty", className="download-btn"),
    ]),
    # Seventh graph with download button
    html.Div([
        dcc.Graph(id="collision-rate-graph"),
        html.Button("Download Rate Penalty", id="btn-download-collision-rate", className="download-btn"),
    ]),

    # Download all button and save as PNG/CSV options
    html.Div([
        html.Button("Download All Charts", id="btn-download-all", className="download-all-btn"),
        html.Div([
            dcc.RadioItems(
                id='download-format',
                options=[
                    {'label': 'PNG Image', 'value': 'png'},
                    {'label': 'CSV Data', 'value': 'csv'}
                ],
                value='png',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            )
        ], style={'margin': '10px 0'})
    ], style={'margin': '20px 0'}),

    # Download component (invisible)
    dcc.Download(id="download-chart"),
    dcc.Download(id="download-all-charts"),

    # Interval for updating
    dcc.Interval(id="interval-component", interval=3000, n_intervals=0)
])


# Update the callback to include exit weight
@app.callback(
    Output("training-loss-graph", "figure"),
    Output("validation-loss-graph", "figure"),
    Output("accuracy-graph", "figure"),
    Output("validation-accuracy-graph", "figure"),
    Output("time-per-step-graph", "figure"),
    Output("exit-weight-graph", "figure"),
    Output("collision-penalty-graph", "figure"),
    Output("collision-rate-graph", "figure"),
    Input("interval-component", "n_intervals")
)
def update_graphs(n):
    df = load_loss_data()
    if df.empty or 'model_name' not in df.columns:
        logging.debug("No training data available yet. Waiting for loss_data.csv...")
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

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

    fig_collision_penalty = px.line(
        df,
        x='epoch',
        y='collision_penalty',
        color='model_name',
        line_dash='model_name',
        symbol='model_name',
        markers=True,
        title="Collision Penalty per Epoch"
    )
    fig_collision_penalty.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Collision Penalty",
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            traceorder="normal")
    )

    fig_collision_rate = px.line(
        df,
        x='epoch',
        y='collision_rate',
        color='model_name',
        line_dash='model_name',
        symbol='model_name',
        markers=True,
        title="Collision Rate per Epoch"
    )
    fig_collision_rate.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Collision Rate",
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            traceorder="normal")
    )

    return fig_training, fig_validation, fig_accuracy, fig_val_accuracy, fig_time_per_step, fig_exit_weight, fig_collision_penalty, fig_collision_rate


# Callback for individual chart downloads
@app.callback(
    Output("download-chart", "data"),
    [
        Input("btn-download-training-loss", "n_clicks"),
        Input("btn-download-validation-loss", "n_clicks"),
        Input("btn-download-accuracy", "n_clicks"),
        Input("btn-download-validation-accuracy", "n_clicks"),
        Input("btn-download-time-per-step", "n_clicks"),
        Input("btn-download-exit-weight", "n_clicks"),
        Input("btn-download-collision-penalty", "n_clicks"),
        Input("btn-download-collision-rate", "n_clicks"),
    ],
    [
        State("training-loss-graph", "figure"),
        State("validation-loss-graph", "figure"),
        State("accuracy-graph", "figure"),
        State("validation-accuracy-graph", "figure"),
        State("time-per-step-graph", "figure"),
        State("exit-weight-graph", "figure"),
        State("collision-penalty-graph", "figure"),
        State("collision-rate-graph", "figure"),
        State("download-format", "value")
    ],
    prevent_initial_call=True
)
def download_chart(n1, n2, n3, n4, n5, n6, n7,
                   fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, format_type):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    charts = {
        "btn-download-training-loss": (fig1, "training_loss"),
        "btn-download-validation-loss": (fig2, "validation_loss"),
        "btn-download-accuracy": (fig3, "accuracy"),
        "btn-download-validation-accuracy": (fig4, "validation_accuracy"),
        "btn-download-time-per-step": (fig5, "time_per_step"),
        "btn-download-exit-weight": (fig6, "exit_weight"),
        "btn-download-collision-penalty": (fig7, "collision_penalty"),
        "btn-download-collision-rate": (fig8, "collision_rate"),
    }

    if button_id not in charts:
        return no_update

    figure, filename_base = charts[button_id]

    if format_type == 'png':
        # For PNG: export the Plotly figure
        img_bytes = BytesIO()
        import plotly.io as pio
        pio.write_image(figure, img_bytes, format='png')
        img_bytes.seek(0)
        return dcc.send_bytes(img_bytes.getvalue(), f"{filename_base}.png")

    elif format_type == 'csv':
        # For CSV: extract data from the figure and create a CSV
        df = load_loss_data()  # Re-load the data
        if df.empty:
            return no_update

        csv_bytes = BytesIO()

        # Map button to column name
        column_mapping = {
            "btn-download-training-loss": "train_loss",
            "btn-download-validation-loss": "val_loss",
            "btn-download-accuracy": "train_acc",
            "btn-download-validation-accuracy": "val_acc",
            "btn-download-time-per-step": "time_per_step",
            "btn-download-exit-weight": "exit_weight",
            "btn-download-collision-penalty": "collision_penalty",
            "btn-download-collision-rate": "collision_rate",
        }

        # Get relevant columns
        column = column_mapping.get(button_id)
        if column:
            relevant_df = df[['model_name', 'epoch', column]].copy()
            relevant_df.to_csv(csv_bytes, index=False)
            csv_bytes.seek(0)
            return dcc.send_bytes(csv_bytes.getvalue(), f"{filename_base}.csv")

    return no_update


# Callback for downloading all charts
@app.callback(
    Output("download-all-charts", "data"),
    Input("btn-download-all", "n_clicks"),
    [
        State("download-format", "value"),
        State("training-loss-graph", "figure"),
        State("validation-loss-graph", "figure"),
        State("accuracy-graph", "figure"),
        State("validation-accuracy-graph", "figure"),
        State("time-per-step-graph", "figure"),
        State("exit-weight-graph", "figure"),
        State("collision-penalty-graph", "figure"),
        State("collision-rate-graph", "figure"),
    ],
    prevent_initial_call=True
)
def download_all_charts(n_clicks, format_type, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
    if not n_clicks:
        return no_update

    if format_type == 'png':
        # For PNG: zip all chart PNGs
        import io
        import zipfile

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            figures = [
                (fig1, "training_loss.png"),
                (fig2, "validation_loss.png"),
                (fig3, "accuracy.png"),
                (fig4, "validation_accuracy.png"),
                (fig5, "time_per_step.png"),
                (fig6, "exit_weight.png"),
                (fig7, "collision_penalty.png"),
                (fig8, "collision_rate.png"),
            ]

            for fig, filename in figures:
                img_bytes = BytesIO()
                import plotly.io as pio
                pio.write_image(fig, img_bytes, format='png')
                img_bytes.seek(0)
                zip_file.writestr(filename, img_bytes.getvalue())

        zip_buffer.seek(0)
        return dcc.send_bytes(zip_buffer.getvalue(), "all_charts.zip")

    elif format_type == 'csv':
        # For CSV: send the complete dataset
        csv_bytes = BytesIO()
        df = load_loss_data()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        return dcc.send_bytes(csv_bytes.getvalue(), "training_data.csv")

    return no_update


if __name__ == '__main__':
    utils.setup_logging()
    app.run(debug=True)  # Changed from app.run_server(debug=True)
