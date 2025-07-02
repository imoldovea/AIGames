import logging
import os
from io import BytesIO

import cv2
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # (Assuming pandas is used in data_to_csv)


def plot_maze_frames(maze, paths, generation, fitnesses=None):
    """
    Create a frame in memory using matplotlib and return it as a NumPy array.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(maze.grid, cmap="binary")

    colors = ['red', 'blue', 'green', 'orange', 'yellow', 'cyan', 'magenta', 'brown', 'pink', 'purple']
    for i, path in enumerate(paths):
        if path:
            px, py = zip(*path)
            color = colors[i % len(colors)]
            label = f'Path {i + 1}' + (f' (fit={fitnesses[i]:.2f})' if fitnesses else '')
            plt.plot(py, px, color=color, linewidth=2, marker='o', markersize=3, label=label)

    plt.title(f"Maze {maze.index} - Generation {generation}")

    # Move the legend to the right of the plot
    plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0., frameon=False)

    plt.axis("off")
    plt.tight_layout()

    # Save the plot to a memory buffer (BytesIO) as a PNG and read it as a NumPy array
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches="tight")  # Adjust bounding box to include the legend
    plt.close()
    buffer.seek(0)

    # FIX: Use imageio.imread(buffer) rather than imageio.v3.imread(buffer)
    img = imageio.imread(buffer)
    buffer.close()
    return img


def create_video_from_memory(frames, output_file="output/evolution.mp4", fps=4):
    """
    Create a video from in-memory NumPy frames with doubled resolution.
    """
    if not frames:
        raise ValueError("No frames provided!")

    # Get the original frame dimensions
    original_height, original_width, _ = frames[0].shape

    # Double the resolution
    doubled_width = original_width * 2
    doubled_height = original_height * 2

    # Initialize the video writer with the new resolution
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (doubled_width, doubled_height))

    for frame in frames:
        # Resize the frame to double its resolution
        resized_frame = cv2.resize(frame, (doubled_width, doubled_height), interpolation=cv2.INTER_LINEAR)
        video.write(cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))  # OpenCV uses BGR format

    # Add the final frame a few times (pause at the end)
    for _ in range(fps * 5):
        final_resized_frame = cv2.resize(frames[-1], (doubled_width, doubled_height), interpolation=cv2.INTER_LINEAR)
        video.write(cv2.cvtColor(final_resized_frame, cv2.COLOR_RGB2BGR))

    video.release()
    logging.debug(f"Video saved as {output_file}")


def create_gif_from_memory(frames, output_gif="output/evolution.gif", duration=0.4):
    """
    Create a GIF from in-memory NumPy frames.
    """
    if not frames:
        raise ValueError("No frames provided!")

    # Get the max shape
    max_height = max(f.shape[0] for f in frames)
    max_width = max(f.shape[1] for f in frames)

    # Pad or resize frames to same size
    padded_frames = []
    for frame in frames:
        h, w, _ = frame.shape
        pad_h = max_height - h
        pad_w = max_width - w
        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad

        padded = np.pad(frame, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant',
                        constant_values=255)
        padded_frames.append(padded)

    imageio.imwrite(output_gif, padded_frames, duration=duration, format="GIF")
    logging.debug(f"GIF saved as {output_gif}")


def data_to_csv(monitoring_data, filename="output/evolution_data.csv"):
    # replace maze with maze.index in monitoring_data
    for data in monitoring_data:
        data["maze_index"] = data["maze"].index
        data["complexity"] = data["maze"].complexity
        data["longest_path"] = max(len(path) for path in data["paths"]) if data["paths"] else 0
        data["generation"] = data["generation"] + 1
        data["avg_fitness"] = round(data["avg_fitness"], 2)
        data["max_fitness"] = round(data["max_fitness"], 2)
        data["diversity"] = round(data["diversity"], 2)
        del data["maze"]
        del data["paths"]

    ordered_columns = [
        "maze_index",
        "complexity",
        "generation",
        "max_fitness",
        "avg_fitness",
        "diversity",
        "longest_path"
    ]

    df = pd.DataFrame(monitoring_data)
    df.sort_values(by=["maze_index", "generation"], inplace=True)
    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)


def visualize_evolution(monitoring_data, mode="video", index=0):
    """
    Visualize the evolution by generating frames in memory and saving a video or GIF.
    """
    frames = []

    for data in monitoring_data:
        frame = plot_maze_frames(
            maze=data["maze"],
            paths=data["paths"],
            generation=data["generation"],
            fitnesses=data.get("fitnesses")
        )
        plt.figure(figsize=(5, 5), dpi=100)  # fixed size
        frames.append(frame)

    os.makedirs("output", exist_ok=True)

    # Save the evolution data in output/evolution_data.csv
    data_to_csv(monitoring_data, "output/evolution_data.csv")

    if mode == "video":
        output_file = f"output/evolution_{index}.mp4"
        create_video_from_memory(frames, output_file, fps=4)
    elif mode == "gif":
        output_gif = f"output/evolution_{index}.gif"
        create_gif_from_memory(frames, output_gif, duration=0.4)


def print_fitness(maze, fitness_history, avg_fitness_history, diversity_history, show=False):
    """
    Plot and save fitness metrics over generations.

    Args:
        fitness_history: Best fitness scores per generation
        avg_fitness_history: Average fitness scores per generation
        diversity_history: Population diversity measures per generation
        show: Whether to display plot interactively
    """
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, label="Best Fitness")
    plt.plot(avg_fitness_history, label="Avg Fitness")
    plt.plot(diversity_history, label="Diversity")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Fitness Over Generations {maze.index}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"output/fitness_plot+{maze.index}.png")  # Save to output directory
    if show:
        plt.show()
    plt.close()  # Always close to release memory
