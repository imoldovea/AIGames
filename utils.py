from datetime import datetime
from fpdf import FPDF
import tempfile
from PIL import Image
import numpy as np
import cv2
import os
import logging
import traceback
import pickle
from multiprocessing import Process

# Define a custom PDF class (optional, for adding a header)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Collection of maze solution', ln=True, align='C')
        self.ln(10)


def save_mazes_as_pdf(solved_mazes: list[str], output_filename: str = "maze_solutions.pdf") -> None:
    """
      Save a collection of maze strings to a PDF file.

      Parameters:
          mazes (list of str): The list of maze representations.
          filename (str): The filename for the output PDF.
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
                # Get the maze image as a numpy array
                image_array = maze_obj.get_maze_as_png(show_path=True, show_solution=True)

                # Save the numpy array as a temporary PNG file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    temp_image_path = tmp_file.name
                Image.fromarray(image_array).save(temp_image_path)

                pdf.add_page()
                # Optionally add a title for each maze
                pdf.set_font("Arial", "B", 16)
                algorithm = maze_obj.get_algorithm()
                pdf.cell(0, 5, f"Algorithm: {algorithm}", ln=True, align='C')
                pdf.ln(5)
                pdf.cell(0, 10, f"Maze {index}", ln=True, align='C')
                pdf.ln(5)
                pdf.cell(0, 15, f"Correct Solution: {maze_obj.test_solution()}", ln=True, align='C')
                pdf.ln(5)
                # Use multi_cell to allow for multi-line maze text
                pdf.image(temp_image_path, x=10, y=30, w=pdf.w - 20)
                os.remove(temp_image_path)
            except Exception as e:
                logging.error(f"An error occurred: {e}\n\nStack Trace:{traceback.format_exc()}")

        # Save the PDF to the specified file
        pdf.output(output_filename)
        logging.info(f"Mazes saved as PDF: {output_filename}")
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


def save_movie(solved_mazes, output_filename="maze_solutions.mp4"):
    """
    Generates and saves a video of all mazes and their solutions using OpenCV,
    with a title screen before each maze and non-overlapping text.
    """
    try:
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'H', '2', '6', '4')  # Custom FourCC for H.264 NVENC
        width, height = 800, 600  # Desired resolution for the final video
        title_frames_count = 10  # Number of frames to show the title screen

        frames = []
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        maze_count = 1

        for maze in solved_mazes:
            try:
                images = maze.get_raw_movie()  # Maze frames as NumPy arrays
                algorithm = maze.get_algorithm()
            except Exception as e:
                logging.warning(f"Could not process maze #{maze_count}: {e}")
                maze_count += 1
                continue

            # 1. CREATE TITLE SCREEN (white background) BEFORE EACH MAZE
            for _ in range(title_frames_count):
                title_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

                # Add text for the title screen
                cv2.putText(title_frame, f"Maze Solver", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
                cv2.putText(title_frame, f"Maze #{maze_count}", (50, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                cv2.putText(title_frame, f"Algorithm: {algorithm}", (50, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                cv2.putText(title_frame, f"Generated on: {now_str}", (50, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                cv2.putText(title_frame, f"Solution: {maze.test_solution()}", (60, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                frames.append(title_frame)

            # 2. ADD MAZE FRAMES WITH A TOP MARGIN FOR TEXT
            margin_height = 60  # pixels reserved at the top for text

            for img in images:
                # Resize img to fit the frame height minus the margin
                desired_height = height - margin_height
                aspect_ratio = img.shape[1] / img.shape[0]
                desired_width = int(desired_height * aspect_ratio)
                resized_img = cv2.resize(img, (desired_width, desired_height), interpolation=cv2.INTER_NEAREST)

                # Create a blank frame with gray background
                frame_with_margin = np.ones((height, width, 3), dtype=np.uint8) * 128

                # Compute horizontal placement
                start_x = (width - resized_img.shape[1]) // 2
                end_x = start_x + resized_img.shape[1]

                # Assign the resized image to the frame
                frame_with_margin[margin_height:height, start_x:end_x] = resized_img

                # Draw white rectangle in the top margin for text
                cv2.rectangle(frame_with_margin, (0, 0), (width, margin_height), (255, 255, 255), -1)

                # Put text in the margin
                cv2.putText(frame_with_margin, f"Maze #{maze_count}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame_with_margin, f"Algorithm: {algorithm}", (250, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame_with_margin, f"Solution: {maze.test_solution()}", (350, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                frames.append(frame_with_margin)

            maze_count += 1

        # Encode the final list of frames into a video (in parallel)
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
