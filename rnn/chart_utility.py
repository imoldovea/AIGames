# chart_utility.py

import cv2
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objs as go
import seaborn as sns
import subprocess
import torch
import torch.onnx
import webbrowser
from PIL import Image
from configparser import ConfigParser
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

from rnn.maze_trainer import load_models

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)
OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
LOSS_DATA = config.get("FILES", "LOSS_DATA", fallback="loss_data.csv")
LOSS_FILE = f"{OUTPUT}{LOSS_DATA}"

import logging

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


# Load the loss data


# Prepare plot with separate lines per model
def save_latest_loss_chart():
    LOSS_CHART = f"{OUTPUT}loss_chart.html"

    if not os.path.exists(LOSS_FILE):
        raise FileNotFoundError(f"Loss file not found: {LOSS_FILE}")

    df = pd.read_csv(LOSS_FILE)
    df['time_per_step'] = (pd.to_numeric(df['time_per_step'], errors='coerce') / 60).round(0)

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[
            "Training Loss", "Validation Loss",
            "Training Accuracy %", "Validation Accuracy %",
            "Time per Step (minutes)"
        ]
    )

    models = df["model_name"].unique()
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']
    metrics = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Val Loss"),
        ("train_acc", "Train Accuracy"),
        ("val_acc", "Val Accuracy"),
        ("time_per_step", "Time per Step")
    ]

    for idx, (metric_key, metric_title) in enumerate(metrics):
        for i, model in enumerate(models):
            model_df = df[df['model_name'] == model]
            fig.add_trace(go.Scatter(
                x=model_df["epoch"],
                y=model_df[metric_key],
                mode="lines+markers",
                name=f"{metric_title} ({model})",
                legendgroup=metric_title,
                marker=dict(symbol=symbols[i % len(symbols)]),
                showlegend=True
            ), row=idx + 1, col=1)

        fig.update_yaxes(title_text=metric_title, row=idx + 1, col=1)

    fig.update_layout(
        height=1400,
        width=1000,
        title_text="Latest Loss Chart (Per Model)",
        legend_tracegroupgap=160,
        legend_title="Model Type"
    )

    fig.write_html(LOSS_CHART, auto_open=False)
    webbrowser.open(f"file://{os.path.abspath(LOSS_CHART)}")

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
                figure = plt.figure(figsize=(16, 12), dpi=300)  # Increased size and DPI
                plt.title(f"Model: {model}", fontsize=14)
                plt.axis("off")
                # png_data = dot.pipe(format="png")

                try:
                    temp_dot_file = os.path.join(output_dir, f"temp_model_{idx}.dot")
                    temp_png_file = os.path.join(output_dir, f"temp_model_{idx}.png")

                    dot.save(temp_dot_file)
                    # Use subprocess to call dot directly
                    subprocess.run([
                        "dot", "-Tpng",
                        "-Gdpi=300",  # Set higher DPI
                        temp_dot_file,
                        "-o", temp_png_file
                    ], check=True, timeout=60)

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

    matplotlib.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
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


def visualize_exit_activations(exit_activations, maze_name="Maze", output_folder=f"{OUTPUT}exit/"):
    """
    Plots the evolution of exit neuron activations over time.

    Args:
        exit_activations (list): A list of sigmoid exit probabilities at each step.
        maze_name (str): Title for the plot.
        output_folder (str): Where to save the output image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(exit_activations, marker='o')
    plt.title(f"Exit Activations Over Time ({maze_name})")
    plt.xlabel("Step")
    plt.ylabel("Exit Probability")
    plt.grid(True)

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"{maze_name}_exit_activations.png")
    plt.savefig(filename)
    plt.close()
    logging.debug(f"Saved exit activations plot to {filename}")


def visualize_exit_confidence_heatmap(confidence_grid, maze_name="Maze", output_folder=f"{OUTPUT}exit/"):
    """
    Generates a heatmap for the exit confidence across maze locations.

    Args:
        confidence_grid (2D array): Exit probabilities per (y,x) position.
        maze_name (str): Title for the heatmap.
        output_folder (str): Where to save the heatmap image.
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(confidence_grid, cmap="YlOrRd", square=True, annot=False, cbar=True)
    plt.title(f"Exit Confidence Map ({maze_name})")
    plt.axis('off')

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"{maze_name}_exit_confidence_heatmap.png")
    plt.savefig(filename)
    plt.close()
    logging.debug(f"Saved exit confidence heatmap to {filename}")


def evaluate_model_success(solver, test_mazes, max_steps=1000):
    """
    Solves test mazes and computes the success rate.

    Args:
        solver: A maze solver object (like RNN2MazeSolver).
        test_mazes (list): List of Maze objects to solve.
        max_steps (int): Maximum steps allowed per maze.

    Returns:
        success_rate (float): Fraction of mazes successfully solved.
    """
    total = len(test_mazes)
    solved = 0

    for maze in test_mazes:
        solver.solve(maze, max_steps=max_steps)
        if maze.at_exit():
            solved += 1

    success_rate = solved / total if total > 0 else 0.0
    logging.info(f"Test Maze Success Rate: {solved}/{total} = {success_rate:.2%}")
    return success_rate


def main():
    save_latest_loss_chart()

    config = ConfigParser()
    config.read("config.properties")
    # Read the allowed models from the config file. Expected format: "GRU, LSTM, RNN"
    models_config = config.get("DEFAULT", "models", fallback="GRU,LSTM,RNN")
    allowed_models = [model.strip().upper() for model in models_config.split(",")]
    models = load_models(allowed_models)
    save_neural_network_diagram(models)
    visualize_model_weights(models)
    # visualize_model_activations(models)


if __name__ == '__main__':
    main()
