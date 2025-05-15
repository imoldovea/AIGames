import logging
import os
import shutil

import cv2
import imageio
import matplotlib.pyplot as plt


def plot_maze_paths(maze, paths, generation, fitnesses=None, out_dir="frames"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.figure(figsize=(5, 5))
    plt.imshow(maze.grid, cmap="binary")
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']  # or cycle more colors
    for i, path in enumerate(paths):
        if path:
            px, py = zip(*path)
            color = colors[i % len(colors)]
            label = f'Path {i + 1}' + (f' (fit={fitnesses[i]:.2f})' if fitnesses else '')
            plt.plot(py, px, color=color, linewidth=2, marker='o', markersize=3, label=label)
    plt.title(f"Maze {maze.index} - Generation {generation}")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    frame_path = os.path.join(out_dir, f"frame_{generation:04d}.png")
    plt.savefig(frame_path)
    plt.close()
    return frame_path


def create_video_from_frames(frames_dir="frames", output_file="output/evolution.mp4", fps=4):
    frames = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')]
    )
    if not frames:
        raise ValueError("No frames found!")
    img = cv2.imread(frames[0])
    height, width, layers = img.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        img = cv2.imread(frame)
        video.write(img)
    for _ in range(fps * 5):
        img = cv2.imread(frames[-1])
        video.write(img)
    video.release()
    logging.debug(f"Video saved as {output_file}")


def create_gif_from_frames(frames_dir="frames", output_gif="output/evolution.gif", duration=0.4):
    frames = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')]
    )
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(output_gif, images, duration=duration)
    logging.debug(f"GIF saved as {output_gif}")


def visualize_evolution(monitoring_data, mode="video", index=0):
    frames_dir = "output/frames"

    # Remove /frames subfolder and its contents if exist (including subdirectories)
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)

    for data in monitoring_data:
        plot_maze_paths(
            maze=data["maze"],
            paths=data["paths"],  # List of paths
            generation=data["generation"],
            fitnesses=data.get("fitnesses"),
            out_dir=frames_dir
        )
    if mode == "video":
        output_file = f"output/evolution_{index}.mp4"
        create_video_from_frames(frames_dir, output_file, fps=4)
    elif mode == "gif":
        output_gif = f"output/evolution_{index}.gif"
        create_gif_from_frames(frames_dir, output_gif, duration=0.4)
    else:
        print("Visualization complete, frames are in 'frames/' directory.")
