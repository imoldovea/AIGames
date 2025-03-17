
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import traceback
import torch.onnx
import io
import torch
from matplotlib.backends.backend_pdf import PdfPages
from configparser import ConfigParser
from torchviz import make_dot
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")


def save_latest_loss_chart(loss_file_path: str, loss_chart: str) -> None:
    """
    Generates and saves the latest training loss and validation loss charts as an HTML file.
    Includes training timings for each epoch (in minutes) and an annotation for total training time per model (in minutes).
    If the first epoch's training time is missing, it is calculated as the average training time of the other epochs.

    Parameters:
    - loss_file_path (str): Path to the CSV file containing loss data.
    - loss_chart (str): Path where the HTML file will be saved.
    """
    logging.debug("Generating latest loss chart...")
    # Start time used for logging, not for annotation

    try:
        # Read the loss data from CSV
        df = pd.read_csv(loss_file_path)
        hover_texts = []

        df = df.copy()
        # Break if the dataset is empty
        if df.empty:
            logging.warning("Dataset is empty. Exiting function.")
            return

        # Validate required columns
        required_columns = {'model', 'epoch', 'loss', 'validation_loss', 'time'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")

        # For each model, fill missing training time in the first epoch (if any)
        models = df['model'].unique()
        for model in models:
            model_mask = df['model'] == model
            # Determine the first epoch for the current model
            first_epoch = df.loc[model_mask, "epoch"].min()
            first_epoch_mask = model_mask & (df["epoch"] == first_epoch)

        # Initialize subplots: 2 rows (training loss and validation loss), 1 column
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Training Loss", "Validation Loss")
        )

        # Plot training loss for each unique model with per-epoch training time (converted from microseconds to minutes)
        for model in models:
            model_df = df[df['model'] == model]
            hover_text = [f"Epoch time: {t/w6e7:.2f} min" for t in model_df["time"]]
            fig.add_trace(
                go.Scatter(
                    x=model_df["epoch"],
                    y=model_df["loss"],
                    mode="lines+markers",
                    name=f"{model} - Loss",
                    text=hover_text,
                    hovertemplate="Epoch: %{x}<br>Loss: %{y}<br>%{text}<extra></extra>"
                ),
                row=1, col=1
            )

        # Plot validation loss for each unique model with per-epoch training time (converted from microseconds to minutes)
        for model in models:
            model_df = df[df['model'] == model]
            hover_text = [f"Epoch time: {t/6e7:.2f} min" for t in model_df["time"]]
            fig.add_trace(
                go.Scatter(
                    x=model_df["epoch"],
                    y=model_df["validation_loss"],
                    mode="lines+markers",
                    name=f"{model} - Validation Loss",
                    text=hover_text,
                    hovertemplate="Epoch: %{x}<br>Validation Loss: %{y}<br>%{text}<extra></extra>"
                ),
                row=2, col=1
            )

        # Compute total training time for each model (using the updated time values) and convert to minutes
        total_time_texts = []
        for model in models:
            total_time = df[df['model'] == model]["time"].sum()
            total_time_minutes = total_time / 6e7
            total_time_texts.append(f"{model}: {total_time_minutes:.2f} min")
        total_training_time_annotation = "Total Training Time per Model - " + ", ".join(total_time_texts)

        # Add an annotation for total training time at the bottom of the chart
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.5, y=-0.2,  # Adjust y-position as needed to avoid overlap
            text=total_training_time_annotation,
            showarrow=False,
            font=dict(size=12, color="black")
        )

        # Update layout for better visualization: set title, axis labels, height, and margins
        fig.update_layout(
            title="Training and Validation Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=800,
            margin=dict(b=200),  # Increase bottom margin to fit annotation
            template=pio.templates.default,
            legend_title="Models"
        )

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
            logging.debug(f"Created directory: {directory}")

        try:
            # Save the figure as an HTML file
            fig.write_html(html_chart)
            # Optionally display the graph
            #fig.show(block=False)
            # Save as PNG image
            image_path = os.path.join(OUTPUT, "loss_chart.png")
            fig.write_image(image_path)
            logging.info(f"Latest loss chart saved as {html_chart} and {image_path}")
        except Exception as e:
            logging.error("Error saving loss chart:")
            logging.error(traceback.format_exc())
    except Exception as e:
        logging.error(f"Error saving latest loss chart: {e}", exc_info=True)
        logging.error(traceback.format_exc())

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
    logging.debug("Generating neural network diagrams...")

    if not models or not isinstance(models, list):
        raise ValueError("The 'models' parameter must be a non-empty list of neural network models.")

    config = ConfigParser()
    config.read("config.properties")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pdf_path = os.path.join(output_dir, "neural_network_diagrams.pdf")

    matplotlib.use("Agg")

    with PdfPages(pdf_path) as pdf:
        for idx, (_, model) in enumerate(models):
            model.eval()  # Set model to evaluation mode
            logging.debug(f"Processing model: {model}")

            input_size = config.getint("DEFAULT", "input_size", fallback=7)
            seq_length = config.getint("DEFAULT", "max_steps", fallback=7)
            batch_size = config.getint("DEFAULT", "batch_size", fallback=7)
            dummy_input = torch.randn(batch_size, seq_length, input_size).to(device)
            output = model(dummy_input)

            # Generate the computational graph using Torchviz and display it using PIL to decode PNG data.
            try:
                dot = make_dot(output, params=dict(model.named_parameters()))

                # Draw the graph and add it as a page in the PDF
                figure = plt.figure(figsize=(12, 8))
                plt.title(f"Model: {getattr(model, 'name', f'Model_{idx}')}")
                plt.axis("off")
                png_data = dot.pipe(format="png")
                pipe_buffer = io.BytesIO(png_data)
                image = Image.open(pipe_buffer)
                plt.imshow(image, aspect="auto")
                pdf.savefig(figure)
                plt.close(figure)

            except Exception as e:
                logging.error(f"Torchviz graph generation failed: {e}")
                raise RuntimeError(f"Torchviz graph generation failed: {e}")

            # Generate ONNX export for the model for Netron visualization.
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
            except Exception as e:
                logging.error(f"ONNX export for Netron visualization failed: {e}")
                raise RuntimeError(f"ONNX export for Netron visualization failed: {e}")

            # Log the model graph for TensorBoard visualization
            try:
                tb_log_dir = os.path.join(output_dir, f"model_{idx}_tensorboard")
                writer = SummaryWriter(log_dir=tb_log_dir)
                writer.add_graph(model, dummy_input)
                writer.close()
            except Exception as e:
                logging.error(f"TensorBoard graph logging failed: {e}")
                raise RuntimeError(f"TensorBoard graph logging failed: {e}")


def visualize_model_weights(models, output_folder=OUTPUT, base_title="Model Diagram", cmap="viridis", **kwargs):
    logging.debug("Visualizing model weights...")
    pdf_filename = f"{output_folder}model_weights_diagrams.pdf"

    with PdfPages(pdf_filename) as pdf:
        for idx, item in enumerate(models):
            # Extract model name and object if item is a tuple, else assume it's the model object.
            if isinstance(item, tuple):
                model_name, model_obj = item
            else:
                model_obj = item
                model_name = getattr(model_obj, "name", f"Model_{idx}")

            weights = model_obj.get_weights()
            if weights:
                # Unpack the first tuple: (parameter_name, parameter_value)
                _, weight_array = weights[0]
                weight_array = np.array(weight_array)  # Convert to NumPy array
                if weight_array.ndim >= 2:
                    scaled_array = np.repeat(weight_array, repeats=50, axis=1)  # Scale dimension 2 (x-axis) by 50
                    data = np.mean(scaled_array, axis=0) if scaled_array.ndim > 2 else scaled_array
                else:
                    data = np.repeat(weight_array, repeats=50, axis=0).reshape(1,
                                                                               -1)  # Scale dimension 1 (x-axis) by 50 and reshape

            plt.figure(figsize=(8, 6))
            plt.imshow(data, cmap=cmap)
            plt.title(f"{model_name} - {base_title}")
            plt.colorbar(label="Weight Value")
            plt.xlabel("Output Neurons")
            plt.ylabel("Input Neurons")

            pdf.savefig()
            plt.close()

    logging.info(f"Saved model diagrams to: {pdf_filename}")

def visualize_model_activations(all_activations, output_folder = OUTPUT, model_name = "Mode Name", video_filename = "recurrent_activations_movie.mp4", fps = 25):
    """
    Generates a video showing model activations over time.

    Args:
        activations (list): A list of numpy arrays representing model activations.
        output_folder (str): Directory where the video will be saved.
        model_name (str): Name of the model (used for labeling output).
        video_filename (str): Name of the output video file.
        fps (int): Frames per second for the video.
    """
    # Ensure the output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Prepare the matplotlib figure.
    fig, ax = plt.subplots()
    frames = []

    for activations in all_activations: #loop in activations for all mazes
        # Loop over activations to generate frames.
        for act in activations:
            # Clear the previous image.
            ax.clear()

            # Squeeze extra dimensions.
            act = np.squeeze(act)

            # If activation is still one-dimensional, reshape to 2D.
            if act.ndim == 1:
                # Option 1: Show as a 1xN heatmap.
                act = act.reshape(1, -1)
                # Option 2: Alternatively, if a square shape is preferred (if possible),
                # you can use:
                # size = int(np.sqrt(act.size))
                # act = act.reshape(size, size)  # ensure the size is valid
            elif act.ndim != 2:
                raise ValueError(f"Activation array has invalid number of dimensions: {act.shape}")

            # Display the activation as a heatmap.
            ax.imshow(act, cmap='viridis')
            ax.set_title(f'{model_name} Activation')
            ax.axis('off')

            # Draw the canvas to update the figure.
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()

            # Retrieve the ARGB buffer and convert it to RGB.
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
            frame = buf[:, :, 1:4].copy()
            frames.append(frame)

    plt.close(fig)

    # Write the frames to a video.
    activations_path = f"{OUTPUT}activations/"
    os.makedirs(activations_path, exist_ok=True)
    video_path = f"{activations_path}{video_filename}"
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        for _ in range(fps * 2):  # Duplicate each frame for 2 seconds
            video_writer.write(frame)

    video_writer.release()
    plt.close('all')
    logging.info(f"Activations Video saved to: {video_path}")

