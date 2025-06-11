# gameoflife_gui.py

import tkinter as tk

import numpy as np

# Define the grid size and cell display size
rows, cols = 20, 20
cell_size = 20  # Size of each cell in pixels

# Define multiple starting patterns
patterns = {
    "Glider": [
        (1, 2), (2, 3), (3, 1), (3, 2), (3, 3)
    ],
    "Small Exploder": [
        (10, 10), (10, 11), (10, 12),
        (9, 11), (11, 11), (9, 10), (9, 12), (11, 10), (11, 12)
    ],
    "Exploder": [
        (5, 5), (5, 6), (5, 7),
        (6, 5), (6, 7),
        (7, 5), (7, 6), (7, 7),
        (8, 5), (8, 7)
    ],
    "Gosper Glider Gun": [
        (5, 1), (5, 2), (6, 1), (6, 2),
        (3, 13), (3, 14), (4, 12), (4, 16), (5, 11), (5, 17),
        (6, 11), (6, 15), (6, 17), (6, 18), (7, 11), (7, 17), (8, 12),
        (8, 16), (9, 13), (9, 14), (4, 21), (5, 21), (6, 21),
        (3, 22), (7, 22), (2, 23), (8, 23), (4, 24), (5, 24), (6, 24),
        (5, 25), (2, 26), (3, 26), (7, 26), (8, 26),
        (4, 27), (5, 27), (6, 27), (5, 28)
    ],
}


class GameOfLife:
    """
    A class to simulate Conway's Game of Life using Tkinter with multiple starting patterns.
    """

    def __init__(self, master):
        """
        Initialize the GUI and set up controls for pattern selection.
        """
        self.master = master

        # Frame for pattern selection buttons
        pattern_frame = tk.Frame(master)
        pattern_frame.pack()

        for pattern_name in patterns:
            btn = tk.Button(pattern_frame, text=pattern_name,
                            command=lambda name=pattern_name: self.set_pattern(name))
            btn.pack(side=tk.LEFT)

        # Create a canvas widget for drawing the grid
        self.canvas = tk.Canvas(master, width=cols * cell_size, height=rows * cell_size)
        self.canvas.pack()

        # Initialize the game board
        self.board = np.zeros((rows, cols), dtype=int)
        self.set_pattern("Glider")  # Default pattern

        # Draw initial state
        self.draw_board()

        # Schedule updates
        self.running = True
        self.master.after(500, self.update)

    def set_pattern(self, pattern_name):
        """
        Set the initial live cells based on selected pattern.
        """
        self.board.fill(0)
        for (r, c) in patterns[pattern_name]:
            if 0 <= r < rows and 0 <= c < cols:
                self.board[r, c] = 1
        self.draw_board()

    def draw_board(self):
        """
        Render the current state of the game board on the Tkinter canvas.
        """
        self.canvas.delete("all")
        for r in range(rows):
            for c in range(cols):
                color = "black" if self.board[r, c] == 1 else "white"
                self.canvas.create_rectangle(
                    c * cell_size, r * cell_size,
                    (c + 1) * cell_size, (r + 1) * cell_size,
                    fill=color, outline="gray"
                )

    def count_neighbors(self, r, c):
        """
        Count live neighbors around a cell.
        """
        total = 0
        for i in range(r - 1, r + 2):
            for j in range(c - 1, c + 2):
                if (i == r and j == c):
                    continue
                if 0 <= i < rows and 0 <= j < cols:
                    total += self.board[i, j]
        return total

    def update(self):
        """
        Compute next generation and refresh display.
        """
        new_board = np.copy(self.board)
        for r in range(rows):
            for c in range(cols):
                neighbors = self.count_neighbors(r, c)
                if self.board[r, c] == 1:
                    if neighbors < 2 or neighbors > 3:
                        new_board[r, c] = 0
                else:
                    if neighbors == 3:
                        new_board[r, c] = 1
        self.board = new_board
        self.draw_board()
        self.master.after(500, self.update)


if __name__ == "__main__":
    """
    Run the Game of Life with options for different starting patterns.
    """
    root = tk.Tk()
    root.title("Game of Life")
    game = GameOfLife(root)
    root.mainloop()
