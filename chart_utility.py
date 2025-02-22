import logging
import os
import pandas as pd
import plotly.graph_objs as go
import traceback
import plotly.io as pio

logging.basicConfig(level=logging.INFO)


def save_latest_loss_chart(loss_file_path, loss_chart):
    """
    Generates and saves the latest training loss chart as an HTML file.

    Parameters:
    - loss_file_path (str): Path to the CSV file containing loss data.
    - loss_chart (str): Path where the HTML file will be saved.
    """
    try:
        # Read the loss data from CSV
        df = pd.read_csv(loss_file_path)

        # Validate required columns
        required_columns = {'model', 'epoch', 'loss'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")

        # Initialize the Plotly figure
        fig = go.Figure()

        # Plot loss for each unique model
        models = df['model'].unique()
        for model in models:
            model_df = df[df['model'] == model]
            fig.add_trace(go.Scatter(
                x=model_df["epoch"],
                y=model_df["loss"],
                mode="lines+markers",
                name=model
            ))

        logging.info("Default Template: %s", pio.templates.default)
        logging.info("Available Templates: %s", list(pio.templates.keys()))

        # Validate loss_chart parameter
        if not loss_chart or not isinstance(loss_chart, str):
            raise ValueError("The 'loss_chart' parameter must be a valid, non-empty string.")

        # Safely change the file extension to .html
        base, ext = os.path.splitext(loss_chart)
        html_chart = f"{base}.html"

        # Ensure the directory exists
        directory = os.path.dirname(html_chart)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")

        try:
            # Customize the layout
            fig.update_layout(
                title="Training Loss Over Time",
            )
            # Save the figure as an HTML file
            fig.write_html(html_chart)
            fig.show()
            logging.info(f"Latest loss chart saved as {html_chart}")
        except Exception as e:
            logging.error("Error saving loss chart:")
            logging.error(traceback.format_exc())

    except Exception as e:
        logging.error("Error saving loss chart:")
        logging.error(traceback.format_exc())
