
import io
import logging
import os
import subprocess
import traceback
from configparser import ConfigParser

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import torch
import torch.onnx
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")



def save_latest_loss_chart(loss_file_path: str, loss_chart: str) -> None:
    """
    Generates and saves separate training loss and validation loss charts as HTML files.
    Includes training timings for each epoch (in minutes) and an annotation for total training time per model (in minutes).
    If the first epoch's training time is missing, it is calculated as the average training time of the other epochs.

    Parameters:
    - loss_file_path (str): Path to the CSV file containing loss data.
    - loss_chart (str): Base path where the HTML files will be saved (training_loss.html and validation_loss.html).
    """
    logging.debug("Generating separate training and validation loss charts...")

    try:
        # Read the loss data from CSV
        df = pd.read_csv(loss_file_path)

        # Break if the dataset is empty
        if df.empty:
            logging.warning("Dataset is empty. Exiting function.")
            return

        # Validate required columns
        required_columns = {'model', 'epoch', 'loss', 'validation_loss', 'time'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")

        # Extract models
        models = df['model'].unique()

        # --- Training Loss Chart ---
        fig_training = go.Figure()

        for model in models:
            model_df = df[df['model'] == model]
            hover_text = [f"Epoch time: {t / 6e7:.2f} min" for t in model_df["time"]]  # Corrected time conversion
            fig_training.add_trace(
                go.Scatter(
                    x=model_df["epoch"],
                    y=model_df["loss"],
                    mode="lines+markers",
                    name=f"{model} - Loss",
                    text=hover_text,
                    hovertemplate="Epoch: %{x}<br>Loss: %{y}<br>%{text}<extra></extra>"
                )
            )

        fig_training.update_layout(
            title="Training Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=800,
            margin=dict(b=150),
            template=pio.templates.default,
            legend_title="Models"
        )

        # --- Validation Loss Chart ---
        fig_validation = go.Figure()

        for model in models:
            model_df = df[df['model'] == model]
            hover_text = [f"Epoch time: {t / 6e7:.2f} min" for t in model_df["time"]]  # Corrected time conversion
            fig_validation.add_trace(
                go.Scatter(
                    x=model_df["epoch"],
                    y=model_df["validation_loss"],
                    mode="lines+markers",
                    name=f"{model} - Validation Loss",
                    text=hover_text,
                    hovertemplate="Epoch: %{x}<br>Validation Loss: %{y}<br>%{text}<extra></extra>"
                )
            )

        fig_validation.update_layout(
            title="Validation Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=800,
            margin=dict(b=150),
            template=pio.templates.default,
            legend_title="Models"
        )

        # --- Save Charts ---
        # Validate loss_chart parameter
        if not loss_chart or not isinstance(loss_chart, str):
            raise ValueError("The 'loss_chart' parameter must be a valid, non-empty string.")

        # Safely change the file extension to .html and create separate filenames
        base, ext = os.path.splitext(loss_chart)
        training_loss_html = f"{base}_training_loss.html"
        validation_loss_html = f"{base}_validation_loss.html"

        # Ensure the directory exists
        directory = os.path.dirname(training_loss_html)  # Use either path; they share the same directory
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logging.debug(f"Created directory: {directory}")

        try:
            # Save the figures as HTML files
            fig_training.write_html(training_loss_html)
            fig_validation.write_html(validation_loss_html)

            # Optionally save as PNG images (adjust paths as needed)
            pio.renderers.default = "png"
            training_loss_image_path = os.path.join(OUTPUT, "training_loss_chart.png")
            validation_loss_image_path = os.path.join(OUTPUT, "validation_loss_chart.png")
            fig_training.write_image(training_loss_image_path)
            fig_validation.write_image(validation_loss_image_path)

            logging.info(
                f"Training loss chart saved as {training_loss_html} and {training_loss_image_path}")
            logging.info(
                f"Validation loss chart saved as {validation_loss_html} and {validation_loss_image_path}")


        except Exception as img_error:
            # Fallback to matplotlib (simplified for brevity)
            print(f"Error saving as PNG: {img_error}")  # removed traceback to reduce tokens
            pass

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
                # png_data = dot.pipe(format="png")

                try:
                    temp_dot_file = os.path.join(output_dir, f"temp_model_{idx}.dot")
                    temp_png_file = os.path.join(output_dir, f"temp_model_{idx}.png")

                    dot.save(temp_dot_file)
                    # Use subprocess to call dot directly
                    subprocess.run(["dot", "-Tpng", temp_dot_file, "-o", temp_png_file],
                                   check=True, timeout=60)

                    # Read the PNG file back
                    with open(temp_png_file, 'rb') as f:
                        png_data = f.read()

                    # Clean up temp files
                    os.remove(temp_dot_file)
                    os.remove(temp_png_file)
                except Exception as e:
                    logging.error(f"Torchviz graph generation failed: {e}")
                    raise RuntimeError(f"Torchviz graph generation failed: {e}")
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


def visualize_model_weights(models, output_folder=OUTPUT, base_title="Model Diagram", cmap="viridis"):
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
        all_activations (list): A list of numpy arrays representing model activations.
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

