import logging
import os
import pandas as pd
import plotly.graph_objs as go
import traceback
import plotly.io as pio

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

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
            # Display the graph in Plots tab (for supported IDEs)
            pio.show(fig)
            fig.show()
            logging.info(f"Latest loss chart saved as {html_chart}")
        except Exception as e:
            logging.error("Error saving loss chart:")
            logging.error(traceback.format_exc())
    except Exception as e:
            logging.error(f"Error saving latest loss chart: {e}", exc_info=True)
            logging.error(traceback.format_exc())

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def save_neural_network_diagram(layer_sizes, output_path="output/neural_network_diagram.png"):
    """
    Draws and saves a neural network diagram with labeled neurons and layers.

    Args:
        layer_sizes: List[int] - number of neurons in each layer.
        labels: List[str] - names of each layer.
        output_path: str - file path to save the diagram image.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Define horizontal spacing of layers
    n_layers = len(layer_sizes)
    v_spacing = 1
    h_spacing = 2
    radius = 0.15

    # Colors per layer (customizable)
    layer_colors = ['gold', 'green', 'red', 'blue', 'purple']
    neuron_positions = []

    # Compute positions for neurons
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2
        positions = [(h_spacing * i, layer_top - v_spacing * j) for j in range(layer_size)]
        neuron_positions.append(positions)

        # Draw neurons
        for x, y in positions:
            circle = Circle((x, y), radius, fill=True, color='white', ec=layer_colors[i % len(layer_colors)], lw=3, zorder=5)
            ax.add_patch(circle)

    # Draw connections
    for idx in range(n_layers - 1):
        for (x1, y1) in neuron_positions[idx]:
            for (x2, y2) in neuron_positions[idx + 1]:
                ax.annotate("", xy=(x2 - radius, y2), xytext=(x1 + radius, y1),
                            arrowprops=dict(arrowstyle="->", color='blue', lw=1))

    plt.title("Neural Network Structure", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Neural network diagram saved at: {output_path}")