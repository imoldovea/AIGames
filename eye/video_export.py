import logging
import os

import cv2

from eye.config import get_display_config
from eye.visualizer import close_pygame, save_frame


def export_movie():
    frame_dir = "output/frames"
    out_file = "output/evolution.mp4"
    fps = get_display_config()['fps']

    # Gather and sort all PNG frame paths:
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not files:
        logging.warning("No frames found in %s; skipping movie export.", frame_dir)
        return

    # Read the first frame to get dimensions:
    first_path = os.path.join(frame_dir, files[0])
    first_frame = cv2.imread(first_path)
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame at {first_path}")

    height, width, channels = first_frame.shape

    # Define the codec and create VideoWriter object.
    # 'mp4v' is a widely supported MPEG-4 codec.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    # Write each frame:
    for fname in files:
        path = os.path.join(frame_dir, fname)
        frame = cv2.imread(path)
        if frame is None:
            logging.warning("Skipping unreadable frame %s", path)
            continue
        writer.write(frame)

    writer.release()
    logging.info(f"Saved MP4 to {out_file}")


def save_fitness_plot(fitness_history):
    os.makedirs("output", exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(fitness_history, color='green')
    plt.title("Fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output/fitness_plot.png")
    plt.close()
    print("Saved fitness plot to output/fitness_plot.png")


def save_last_frame():
    frame_dir = "output/frames"
    out_dir = "output"

    # Find the last frame
    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    if not files:
        logging.warning("No frames found in %s", frame_dir)
        return

    # Copy the last frame
    last_frame = files[-1]
    source = os.path.join(frame_dir, last_frame)
    destination = os.path.join(out_dir, "final_frame.png")
    import shutil
    shutil.copy2(source, destination)
    logging.info(f"Saved final frame to {destination}")


def run_ga():
    export_movie()

    # save the final rendered frame as an image
    save_frame(screen, gen)
    close_pygame()
