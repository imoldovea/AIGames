# chart_utility.py

import io
import os
import subprocess
from configparser import ConfigParser

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import psutil
import torch
import torch.onnx
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

PARAMETERS_FILE = "../config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
LOSS_FILE = f"{OUTPUT}loss_data.csv"

import pandas as pd
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def save_latest_loss_chart(output_file_name=f"{OUTPUT}latest_loss_chart.html"):
    # Ensure epoch is numeric (if needed)
    try:
        loss_data = pd.read_csv(LOSS_FILE)
        loss_data['epoch'] = pd.to_numeric(loss_data['epoch'], errors='coerce')
    except FileNotFoundError:
        logging.error(f"Loss file not found: {LOSS_FILE}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Loss file is empty: {LOSS_FILE}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the loss data: {e}")
        raise Exception(f"An error occurred while loading the loss data: {e}")

    memory_info = psutil.virtual_memory()

    # Prepare aggregate metrics over epochs
    training_loss = loss_data.groupby('epoch')['training_loss'].mean()
    validation_loss = loss_data.groupby('epoch')['validation_loss'].mean()
    accuracy = loss_data.groupby('epoch')['accuracy'].mean()
    validation_accuracy = loss_data.groupby('epoch')['validation_accuracy'].mean()
    time_per_step_avg = (loss_data.groupby('epoch')['time_per_step'].mean() / 60).round(0)

    # Create a figure with 6 vertical subplots (separated Time per Step)
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Training Loss",
            "Validation Loss",
            "Accuracy",
            "Validation Accuracy",
            "Time per Step",
            "Resource Usage"
        )
    )

    # Chart 1: Training Loss
    fig.add_trace(
        go.Scatter(x=training_loss.index, y=training_loss.values, mode='lines+markers', name="Training Loss"),
        row=1, col=1
    )

    # Chart 2: Validation Loss
    fig.add_trace(
        go.Scatter(x=validation_loss.index, y=validation_loss.values, mode='lines+markers', name="Validation Loss",
                   marker=dict(color='orange')),
        row=2, col=1
    )

    # Chart 3: Accuracy
    fig.add_trace(
        go.Scatter(x=accuracy.index, y=accuracy.values, mode='lines+markers', name="Accuracy",
                   marker=dict(color='green')),
        row=3, col=1
    )

    # Chart 4: Validation Accuracy
    fig.add_trace(
        go.Scatter(x=validation_accuracy.index, y=validation_accuracy.values, mode='lines+markers',
                   name="Validation Accuracy", marker=dict(color='red')),
        row=4, col=1
    )

    # Chart 5: Time per Step (dedicated chart)
    models = loss_data['model'].unique()
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Add average time per step
    fig.add_trace(
        go.Scatter(
            x=time_per_step_avg.index,
            y=time_per_step_avg.values,
            mode='lines+markers',
            name="Time per Step (avg)",
            marker=dict(symbol='diamond', color='blue')
        ),
        row=5, col=1
    )

    # Add individual model traces for time per step
    for i, model in enumerate(models):
        model_data = loss_data[loss_data['model'] == model]
        time_per_step_sec = model_data['time_per_step']
        # Convert time per step from seconds to minutes
        time_per_step_min = (time_per_step_sec / 60.0).round(0)

        fig.add_trace(
            go.Scatter(
                x=model_data['epoch'],
                y=time_per_step_min,
                mode='lines+markers',
                name=f"Time per Step (minutes) ({model})",
                marker=dict(color=colors[i % len(colors)])
            ),
            row=5, col=1
        )

    # Chart 6: Resource Usage
    agg_cols = ['cpu_load', 'gpu_load', 'ram_usage']
    resource_avg = loss_data.groupby('epoch')[agg_cols].mean()

    fig.add_trace(
        go.Scatter(
            x=resource_avg.index,
            y=resource_avg['cpu_load'],
            mode='lines+markers',
            name="CPU Load (avg)",
            marker=dict(symbol='diamond', color='magenta')
        ),
        row=6, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=resource_avg.index,
            y=resource_avg['gpu_load'],
            mode='lines+markers',
            name="GPU Load (avg)",
            marker=dict(symbol='diamond', color='cyan')
        ),
        row=6, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=resource_avg.index,
            y=(resource_avg['ram_usage'] / memory_info.total) * 100,
            mode='lines+markers',
            name="RAM Usage (avg)",
            marker=dict(symbol='diamond', color='brown')
        ),
        row=6, col=1
    )
    fig.update_yaxes(range=[0, 100], row=6, col=1)

    # Disable the global legend
    fig.update_layout(
        showlegend=False,
        height=1400,  # Increased height for 6 charts
        width=900,
        title_text="Latest Loss Chart",
        margin=dict(b=150)
    )

    # Custom annotations - updated for 6 charts
    annotations = [
        dict(
            x=0.98, y=0.97, xref="paper", yref="paper",
            text="<b>Training Loss</b>", showarrow=False,
            font=dict(size=10, color="black"),
            xanchor="right", yanchor="top"
        ),
        dict(
            x=0.98, y=0.81, xref="paper", yref="paper",
            text="<b>Validation Loss</b>", showarrow=False,
            font=dict(size=10, color="black"),
            xanchor="right", yanchor="top"
        ),
        dict(
            x=0.98, y=0.65, xref="paper", yref="paper",
            text="<b>Accuracy</b>", showarrow=False,
            font=dict(size=10, color="black"),
            xanchor="right", yanchor="top"
        ),
        dict(
            x=0.98, y=0.49, xref="paper", yref="paper",
            text="<b>Validation Accuracy</b>", showarrow=False,
            font=dict(size=10, color="black"),
            xanchor="right", yanchor="top"
        ),
        dict(
            x=0.98, y=0.33, xref="paper", yref="paper",
            text="<b>Time per Step</b>", showarrow=False,
            font=dict(size=10, color="black"),
            xanchor="right", yanchor="top"
        ),
        dict(
            x=0.98, y=0.17, xref="paper", yref="paper",
            text="<b>Resource Usage</b>", showarrow=False,
            font=dict(size=10, color="black"),
            xanchor="right", yanchor="top"
        ),
        dict(
            x=1.05, y=0.10, xref="paper", yref="paper",
            text=(
                "<b>Legend:</b><br>"
                "<span style='color:magenta'>• Diamond: CPU Load (avg)</span><br>"
                "<span style='color:cyan'>• Diamond: GPU Load (avg)</span><br>"
                "<span style='color:brown'>• Diamond: RAM Usage (avg)</span>"
            ),
            showarrow=False,
            font=dict(size=10, color="black"),
            xanchor="left", yanchor="bottom"
        )
    ]
    fig.update_layout(
        width=1200,  # Increase this value as necessary
        margin=dict(r=250)  # Increase right margin if required
    )
    fig.update_layout(annotations=annotations)

    # Restore full-width for the last two charts
    fig.update_xaxes(domain=[0.0, 1.0], row=5, col=1)
    fig.update_xaxes(domain=[0.0, 1.0], row=6, col=1)

    # Save to HTML
    pio.write_html(fig, file=output_file_name, auto_open=True)

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
            # Move model to appropriate device (GPU or CPU)
            model = model.to(device)

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


def main():
    save_latest_loss_chart()


if __name__ == '__main__':
    main()
