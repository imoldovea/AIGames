import os
import numpy as np
from backtrack_maze_solver import BacktrackingMazeSolver
from bfs_maze_solver import BFSMazeSolver
from maze import Maze
from fpdf import FPDF
import tempfile
from PIL import Image
import logging
from datetime import datetime
from line_profiler import LineProfiler
import cProfile
import pstats
import io
import cv2
from multiprocessing import Process


OUTPUT_MOVIE_FILE = "output/maze_animation.mp4"
logging.basicConfig(level=logging.WARN)

# Define a custom PDF class (optional, for adding a header)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Collection of maze solution', ln=True, align='C')
        self.ln(10)

def load_mazes(file_path):
    """
    Loads mazes from a numpy file.

    Args:
        file_path (str): Path to the numpy file containing mazes.

    Returns:
        list: A list of maze matrices.
    """
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise FileNotFoundError(f"Could not load mazes from {file_path}: {e}")


def solve_all_mazes(mazes, solver_class):
    """
    Solves all the mazes using the provided solver class.

    Args:
        mazes (list): List of maze matrices.
        solver_class (class): Solver class to solve the mazes.

    Returns:
        list: List of tuples with the maze and its solution path.
    """
    solved_mazes = []
    for i, maze_matrix in enumerate(mazes):
        maze_obj = Maze(maze_matrix)
        maze_obj.set_save_movie(True)
        solver = solver_class(maze_obj)

        try:
            solution = solver.solve()
            maze_obj.set_solution(solution)
            solved_mazes.append((maze_obj, solution))
            logging.debug(f"Maze {i + 1} solved successfully.")
        except Exception as e:
            solved_mazes.append((maze_obj, None))
            logging.error(f"Failed to solve maze {i + 1}: {e}")
    return solved_mazes


def save_mazes_as_pdf(solved_mazes, output_filename="maze_solutions.pdf"):
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
        for index, (maze_obj, solution) in enumerate(solved_mazes, start=1):
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
                # Use multi_cell to allow for multi-line maze text
                pdf.image(temp_image_path, x=10, y=30, w=pdf.w - 20)
                os.remove(temp_image_path)
            except Exception as e:
                logging.error(f"Error processing maze {index}: {e}")

        # Save the PDF to the specified file
        pdf.output(output_filename)
        logging.info(f"Mazes saved as PDF: {output_filename}")
    except Exception as e:
        logging.error(f"Failed to save mazes to PDF: {e}")


def display_all_mazes(solved_mazes):
    """
    Displays all the mazes and their solutions.

    Args:
        solved_mazes (list): List of tuples with the maze and its solution.
    """
    for i, (maze, solution) in enumerate(solved_mazes):
        try:
            logging.debug(f"Displaying maze {i + 1}...")
            maze.plot_maze(show_path=False, show_solution=True)
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
    fps = 10
    width, height = 800, 600  # Desired resolution for the final video
    title_frames_count = 10  # Number of frames to show the title screen

    frames = []
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    maze_count = 1

    for maze, solution in solved_mazes:
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

            frames.append(frame_with_margin)

        maze_count += 1

    # Encode the final list of frames into a video (in parallel)
    p = Process(target=encode_video, args=(frames, output_filename, fps, width, height))
    p.start()
    p.join()

    logging.info("Video encoding completed.")

def main():
    """
    Main function to load, solve, and save all mazes into a PDF.
    """
    input_file = "input/mazes.npy"
    output_pdf = "output/solved_mazes.pdf"
    output_mp4 = "output/solved_mazes.mp4"
    try:
        # Step 1: Load mazes
        mazes = load_mazes(input_file)[:1]

        s = io.StringIO()
        pr = cProfile.Profile()

        # Step 2: Solve
        pr.enable()
        solved_mazes_backtrack = solve_all_mazes(mazes, BacktrackingMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        logging.info(f"Backtracking execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms

        pr.enable()
        solved_mazes_bfs = solve_all_mazes(mazes, BFSMazeSolver)
        pr.disable()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        logging.info(f"BFS execution time: {ps.total_tt * 1_000:.2f} ms")  # Convert seconds to ms

        # Step 3: Save mazes to PDF
        solved_mazes = solved_mazes_backtrack + solved_mazes_bfs
        save_mazes_as_pdf(solved_mazes, output_pdf)
        display_all_mazes(solved_mazes)

        lp = LineProfiler()
        lp.add_function(save_movie)
        lp.enable()
        save_movie(solved_mazes, output_mp4)
        lp.disable()
        lp.print_stats()

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
