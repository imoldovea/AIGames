import logging
import os
import pickle
import shutil
import tempfile
import traceback
from configparser import ConfigParser
from datetime import datetime
from multiprocessing import Process

import cv2
import flask.cli
import numpy as np
from PIL import Image
from fpdf import FPDF

PARAMETERS_FILE = "config.properties"
config = ConfigParser()
config.read(PARAMETERS_FILE)

OUTPUT = config.get("FILES", "OUTPUT", fallback="output/")
BRIGHT_PINK = (255, 0, 255)  # Bright pink/magenta color (B,G,R format)


class CustomLogFilter(logging.Filter):
    def __init__(self, forbidden_substrings=None):
        super().__init__()
        # forbidden_substrings is a list of strings that, if found in a log message,
        # will cause the message to be filtered out.
        self.forbidden_substrings = forbidden_substrings or []

    def filter(self, record):
        message = record.getMessage().lower()
        # Return False (filter out) if any forbidden substring is found in the log message.
        for substring in self.forbidden_substrings:
            if substring.lower() in message:
                return False
        return True


def setup_logging():
    logger = logging.getLogger()

    # Remove any existing handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.CRITICAL)
    werkzeug_logger.disabled = True
    werkzeug_logger.propagate = False

    flask.cli.show_server_banner = lambda *args, **kwargs: None  # Suppress Flask's startup messages
    logging.getLogger('werkzeug').setLevel(logging.CRITICAL)

    logger.setLevel(logging.DEBUG)  # Capture all levels of logs

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    # Add more substrings as needed
    forbidden_logs = [
        "findfont", "werkzeug", "werkzeug:_internal.py", "dash-update-component",
        "internal.py", "pydevd", "TF_ENABLE_ONEDNN_OPTS",
        "Training batch", "tqdm"  # Added these two patterns
    ]

    # Console handler for INFO level and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CustomLogFilter(forbidden_logs))
    logger.addHandler(console_handler)

    # File handler for DEBUG level and above
    file_handler = logging.FileHandler(f"{OUTPUT}debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(CustomLogFilter(forbidden_logs))
    logger.addHandler(file_handler)


    #Delete all OUTPUT folder content
    if config.getboolean("DEFAULT", "retrain_model", fallback=True):
        shutil.rmtree(OUTPUT, ignore_errors=True)
        os.makedirs(OUTPUT, exist_ok=True)
        logging.info(f"{OUTPUT}cleared...")

# Define a custom PDF class (optional, for adding a header)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Collection of maze solution', ln=True, align='C')
        self.ln(10)


def save_mazes_as_pdf(solved_mazes: list[str], output_filename: str = "maze_solutions.pdf") -> None:
    """
      Save a collection of maze strings to a PDF file`.

      Parameters:
          solved_mazes (list of str): The list of maze representations.
          output_filename (str): The filename for the output PDF.
      """
    # Generate the title and subtitle
    title = "Maze Solution"
    date = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        # Set a base font for the document
        pdf.set_font("Arial", size=12)

        # Add a cover page with title, algorithm, and date
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 15, title, ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, date, ln=True, align='C')
        pdf.ln(20)

        # Iterate over each maze in the list
        for index, maze_obj in enumerate(solved_mazes, start=1):
            try:
                valid_solution = maze_obj.test_solution()
                # Retrieve the list of solution steps
                solution_steps = maze_obj.get_solution()
                # Calculate the number of steps in the solution
                number_of_steps = len(solution_steps)

                # Get the maze image as a numpy array
                image_array = maze_obj.get_maze_as_png(show_path=True, show_solution=True)

                # Save the numpy array as a temporary PNG file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    temp_image_path = tmp_file.name
                Image.fromarray(image_array).save(temp_image_path)

                pdf.add_page()
                # Optionally add a title for each maze
                pdf.set_font("Arial", "B", 16)
                algorithm = maze_obj.algorithm

                pdf.cell(0, 5, f"Algorithm: {algorithm}", ln=True, align='C')
                pdf.set_text_color(0, 0, 0)  # Reset text color to black after
                pdf.ln(5)
                pdf.cell(0, 10, f"Maze: {index}", ln=True, align='C')
                pdf.ln(5)
                if valid_solution:
                    pdf.set_text_color(0, 128, 0)  # Green for valid solution
                else:
                    pdf.set_text_color(255, 0, 0)  # Red for invalid solution
                pdf.cell(0, 15, f"Correct Solution: {valid_solution}", ln=True, align='C')
                pdf.ln(5)
                pdf.cell(0, 15, f"Solution Steps: {number_of_steps}", ln=True, align='C')
                pdf.ln(5)
                # Use multi_cell to allow for multi-line maze text
                pdf.image(temp_image_path, x=10, y=pdf.get_y(), w=pdf.w - 20)

                os.remove(temp_image_path)
            except Exception as e:
                logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")

        # Save the PDF to the specified file
        pdf.output(output_filename)
        logging.debug(f"Mazes saved as PDF: {output_filename}")
    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")


def display_all_mazes(solved_mazes: list) -> None:
    """
    Displays all the mazes and their solutions.

    Args:
        solved_mazes (list): List of tuples with the maze and its solution.
    """
    for i, maze in enumerate(solved_mazes):
        try:
            logging.debug(f"Displaying maze {i + 1}...")
            maze.plot_maze(show_path=False, show_solution=True,show_position=False)
        except Exception as e:
            logging.warning(f"Could not display maze {i + 1}: {e}")


def encode_video(frame_list, filename, fps_val, w, h):
    """
    Encodes the frames into a video file using OpenCV.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps_val, (w, h))

    for frame in frame_list:
        out.write(frame)

    out.release()
    logging.info(f"Video saved as: {filename}")


def save_movie(solved_mazes, output_filename="output/maze_solutions.mp4"):
    """
    Generates and saves a video of all mazes and their solutions using OpenCV.
    Highlights the current position in the maze solution process.
    """
    logging.info("Generating solution video...")
    try:
        fps = 5
        width, height = 800, 600  # Desired resolution for the final video
        title_frames_count = 10  # Number of frames to show the title screen

        frames = []
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        maze_count = 1

        for maze in solved_mazes:
            try:
                # Retrieve maze frames and current positions
                # images = maze.get_raw_movie()  # Maze frames as NumPy arrays
                images = maze.get_frames()  # Online generation of frames, highlighting the current postion.
                algorithm = maze.algorithm

                # Current positions (list of tuples per frame, e.g., [(x1, y1), (x2, y2)...])
                current_positions = maze.get_current_positions() if hasattr(maze, 'get_current_positions') else None
            except Exception as e:
                logging.warning(f"Could not process maze #{maze_count}: {e}")
                maze_count += 1
                continue

            # 1. CREATE TITLE SCREEN (as in the original code)
            for _ in range(title_frames_count):
                title_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

                cv2.putText(title_frame, f"Maze Solver", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
                cv2.putText(title_frame, f"Maze #{maze_count}", (50, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                cv2.putText(title_frame, f"Algorithm: {algorithm}", (50, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                cv2.putText(title_frame, f"Generated on: {now_str}", (50, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                frames.append(title_frame)

            # 2. PROCESS MAZE FRAMES
            margin_height = 60  # pixels reserved at the top for text

            for idx, img in enumerate(images):
                # Resize image to fit the frame height minus the margin
                desired_height = height - margin_height
                aspect_ratio = img.shape[1] / img.shape[0]
                desired_width = int(desired_height * aspect_ratio)
                resized_img = cv2.resize(img, (desired_width, desired_height), interpolation=cv2.INTER_NEAREST)

                # Create a blank frame with gray background
                frame_with_margin = np.ones((height, width, 3), dtype=np.uint8) * 128

                # Compute position for centered placement
                start_x = (width - resized_img.shape[1]) // 2
                end_x = start_x + resized_img.shape[1]

                # Assign resized image to the frame
                frame_with_margin[margin_height:height, start_x:end_x] = resized_img

                # Highlight the current position (if available)
                if current_positions and idx < len(current_positions):
                    current_pos = current_positions[idx]  # Retrieve current position for this frame

                    if current_pos:
                        cx, cy = current_pos
                        # Adjust position for resizing or margins
                        px = start_x + int(cy * (resized_img.shape[1] / maze.cols))
                        py = margin_height + int(cx * (resized_img.shape[0] / maze.rows))
                        cv2.circle(frame_with_margin, (px, py), radius=10, color=BRIGHT_PINK, thickness=-1)

                # Draw white rectangle for margin text
                cv2.rectangle(frame_with_margin, (0, 0), (width, margin_height), (255, 255, 255), -1)

                # Insert text in the margin
                cv2.putText(frame_with_margin, f"Maze #{maze_count}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame_with_margin, f"Algorithm: {algorithm}", (250, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                last_frame = frame_with_margin
                frames.append(frame_with_margin)
            maze_count += 1

        # 3. ADD PAUSE AFTER LAST FRAME
        pause_frame_count = fps * 3
        for _ in range(pause_frame_count):
            frames.append(last_frame)

        # 4. ENCODE TO VIDEO
        p = Process(target=encode_video, args=(frames, output_filename, fps, width, height))
        p.start()
        p.join()

        logging.info("Video encoding completed.")
    except Exception as e:
        logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")


def load_mazes(file_path = "input/mazes.pkl"):
    """
    Loads mazes from a numpy file.

    Args:
        file_path (str): Path to the numpy file containing mazes.

    Returns:
        list: A list of maze matrices.
    """
    try:
        with open(file_path, 'rb') as f:
            mazes = pickle.load(f)
        logging.info(f"Loaded {len(mazes)} mazes.")
        return mazes
    except Exception as e:
        raise FileNotFoundError(f"Could not load mazes from {file_path}: {e}")
