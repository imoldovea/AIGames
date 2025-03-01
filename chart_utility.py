import logging
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px

OUTPUT = "output/"

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
            fig.write_image(f"{OUTPUT}loss_chart.png")
            logging.info(f"Latest loss chart saved as {html_chart}")
        except Exception as e:
            logging.error("Error saving loss chart:")
            logging.error(traceback.format_exc())
    except Exception as e:
            logging.error(f"Error saving latest loss chart: {e}", exc_info=True)
            logging.error(traceback.format_exc())


import os
import torch
from torchviz import make_dot
import traceback
from torch.utils.tensorboard import SummaryWriter
import torch.onnx
import logging

def save_neural_network_diagram(models, output_dir="output/"):
    """
    Draws and saves neural network diagrams using Torchviz,
    exports to ONNX for Netron visualization, and logs the graph for TensorBoard.

    Parameters:
    - models (list): List of models (nn.Module) for which to generate diagrams.
    - output_dir (str): Directory where diagram images and logs will be saved.

    Raises:
    - ValueError: If `models` is empty or not a list.
    - RuntimeError: If graph generation, export, or file writing fails.
    """
    if not models or not isinstance(models, list):
        raise ValueError("The 'models' parameter must be a non-empty list of neural network models.")

    os.makedirs(output_dir, exist_ok=True)

    for idx, model_tuple in enumerate(models):
        model = model_tuple[1]  # Adjust the index based on your tuple structure
        model.eval()  # Set model to evaluation mode
        print(model)

        input_size = 4
        seq_length = 20
        batch_size = 1
        dummy_input = torch.randn(batch_size, seq_length, input_size)  # Fixed batch size of 1
        output = model(dummy_input)

        ## 1. Torchviz: Generate a PDF of the computational graph.
        try:
            dot = make_dot(output, params=dict(model.named_parameters()))
            torchviz_path = os.path.join(output_dir, f"model_{idx}_torchviz")
            dot.format = "pdf"
            dot.render(torchviz_path, cleanup=True)
        except Exception as e:
            logging.error(f"Torchviz graph generation failed: {e}")
            raise RuntimeError(f"Torchviz graph generation failed: {e}")

        ## 2. ONNX: Export the model to ONNX format for Netron visualization.
        try:
            onnx_path = os.path.join(output_dir, f"model_{idx}.onnx")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 1: 'sequence_length'},
                    'output': {0: 'batch_size'}
                }
            )
            # Open the ONNX file in Netron (manually or via netron.start(onnx_path) for interactive visualization)
        except Exception as e:
            logging.error(f"ONNX export for Netron visualization failed: {e}")
            raise RuntimeError(f"ONNX export for Netron visualization failed: {e}")

        ## 3. TensorBoard: Log the model graph for visualization.
        try:
            tb_log_dir = os.path.join(output_dir, f"model_{idx}_tensorboard")
            writer = SummaryWriter(log_dir=tb_log_dir)
            writer.add_graph(model, dummy_input)
            writer.close()
            # To view the graph, run: tensorboard --logdir={tb_log_dir} in your terminal.
        except Exception as e:
            logging.error(f"TensorBoard graph logging failed: {e}")
            raise RuntimeError(f"TensorBoard graph logging failed: {e}")


def visualize_model_weights(models):
    """
    Plots a histogram of the weights for each trainable parameter in the model.
    Each parameter's weight distribution is displayed in its own Plotly figure.
    """
    for idx, model_tuple in enumerate(models):
        model = model_tuple[1]  # Adjust the index based on your tuple structure
        model.eval()  # Set model to evaluation mode
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Flatten the weights into a 1D array
                weights = param.detach().cpu().numpy().flatten()

                # Create a histogram of the weights
                fig = px.histogram(weights, nbins=30, title=f"Weights Distribution: {name}")
                fig.update_layout(xaxis_title="Weight Value", yaxis_title="Frequency")
                fig.show()
                #fig.write_image(f"{OUTPUT}model_{name}_weights_{name}.png")