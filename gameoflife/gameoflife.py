# gameoflife_gui.py

import tkinter as tk

import numpy as np

rows, cols = 80, 120
cell_size = 12

patterns = {
    "Glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "Gosper Glider Gun": [
        (5, 1), (5, 2), (6, 1), (6, 2), (5, 11), (6, 11), (7, 11), (4, 12), (8, 12),
        (3, 13), (9, 13), (3, 14), (9, 14), (6, 15), (4, 16), (8, 16), (5, 17),
        (6, 17), (7, 17), (6, 18), (3, 21), (4, 21), (5, 21), (3, 22), (4, 22),
        (5, 22), (2, 23), (6, 23), (1, 25), (2, 25), (6, 25), (7, 25),
        (3, 35), (4, 35), (3, 36), (4, 36)
    ],
    "R-pentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "Diehard": [(0, 6), (1, 0), (1, 1), (2, 1), (2, 5), (2, 6), (2, 7)],
    "Acorn": [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)],
    "Lightweight Spaceship": [(0, 1), (0, 4), (1, 0), (2, 0), (3, 0), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)],
    "Middleweight Spaceship": [(0, 2), (0, 5), (1, 0), (2, 0), (3, 0), (4, 0), (4, 5), (5, 0), (5, 1), (5, 2), (5, 3),
                               (5, 4)],
    "Heavyweight Spaceship": [(0, 2), (0, 6), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 6), (6, 0), (6, 1), (6, 2),
                              (6, 3), (6, 4), (6, 5)],
    "Blinker": [(0, 1), (1, 1), (2, 1)],
    "Toad": [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
    "Beacon": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)],
    "Pulsar": [
        (0, 2), (0, 3), (0, 4), (0, 8), (0, 9), (0, 10),
        (2, 0), (2, 5), (2, 7), (2, 12),
        (3, 0), (3, 5), (3, 7), (3, 12),
        (4, 0), (4, 5), (4, 7), (4, 12),
        (5, 2), (5, 3), (5, 4), (5, 8), (5, 9), (5, 10),
        (7, 2), (7, 3), (7, 4), (7, 8), (7, 9), (7, 10),
        (8, 0), (8, 5), (8, 7), (8, 12),
        (9, 0), (9, 5), (9, 7), (9, 12),
        (10, 0), (10, 5), (10, 7), (10, 12),
        (12, 2), (12, 3), (12, 4), (12, 8), (12, 9), (12, 10),
    ],
    "Puffer train": [
        (0, 2), (0, 3), (1, 1), (1, 4), (2, 1), (2, 4), (3, 2), (3, 3),
        (8, 2), (8, 3), (9, 1), (9, 4), (10, 1), (10, 4), (11, 2), (11, 3),
    ],
    "Switch engine": [(0, 1), (1, 3), (2, 0), (2, 1), (3, 2), (3, 3), (3, 4)],
    "Rake": [
        (10, 10), (10, 11), (11, 10), (11, 11),
        (10, 20), (11, 20), (12, 20), (9, 21), (13, 21), (8, 22), (14, 22), (8, 23), (14, 23),
        (11, 24), (9, 25), (13, 25), (10, 26), (11, 26), (12, 26), (11, 27)
    ],
    "Spacefiller": [
        (0, 2), (0, 4), (1, 0), (1, 1), (1, 5), (2, 0), (2, 2), (2, 3), (2, 5),
        (3, 0), (3, 1), (3, 5), (4, 3), (4, 5), (5, 0), (5, 2), (5, 4), (6, 0),
        (6, 1), (6, 3)
    ],
    "Cordership": [
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
        (4, 0), (4, 7), (5, 0), (5, 7), (6, 1), (6, 2), (6, 5), (6, 6)
    ],
    "Logic gate circuits": [
        # NOT gate
        (5, 5), (5, 6), (6, 5), (6, 7), (7, 5), (8, 10), (9, 9), (9, 11), (10, 10),
        # AND gate
        (15, 5), (15, 6), (16, 5), (16, 7), (17, 5), (15, 10), (16, 10), (17, 10), (17, 11),
        (18, 9), (18, 12), (19, 9), (19, 12), (20, 10), (20, 11)
    ],
    "Binary counter": [
        (2, 13), (3, 11), (3, 13), (4, 12), (5, 12), (6, 12), (9, 11), (9, 12), (9, 13),
        (10, 11), (10, 12), (10, 13), (11, 11), (11, 12), (11, 13),
        (13, 11), (13, 12), (13, 13), (14, 11), (14, 12), (14, 13),
        (15, 11), (15, 12), (15, 13)
    ],
    "Breeder": [
        (5, 1), (6, 1), (5, 2), (6, 2),
        (5, 10), (6, 10), (7, 10), (4, 11), (8, 11), (3, 12), (9, 12), (3, 13), (9, 13),
        (6, 14), (4, 15), (8, 15), (5, 16), (6, 16), (7, 16),
        (6, 17), (6, 22), (7, 22), (8, 22), (6, 23), (7, 23), (8, 23),
        (6, 24), (7, 24), (8, 24)
    ]
}


class GameOfLife:
    def __init__(self, master):
        self.master = master
        self.is_running = False

        # Pattern menu
        menubar = tk.Menu(master)
        master.config(menu=menubar)
        pattern_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Patterns", menu=pattern_menu)
        for pattern_name in sorted(list(patterns.keys())):
            pattern_menu.add_command(label=pattern_name,
                                     command=lambda name=pattern_name: self.set_pattern(name))

        # Control buttons
        control_frame = tk.Frame(master)
        control_frame.pack()

        self.play_pause_btn = tk.Button(control_frame, text="Play", command=self.toggle_run)
        self.play_pause_btn.pack(side=tk.LEFT)

        self.next_btn = tk.Button(control_frame, text="Next Step", command=self.next_step)
        self.next_btn.pack(side=tk.LEFT)

        self.clear_btn = tk.Button(control_frame, text="Clear", command=self.clear_board)
        self.clear_btn.pack(side=tk.LEFT)

        # Canvas
        self.canvas = tk.Canvas(master, width=cols * cell_size, height=rows * cell_size)
        self.canvas.pack()

        # Create cell grid for drawing
        self.cells = np.empty((rows, cols), dtype=object)
        for r in range(rows):
            for c in range(cols):
                self.cells[r, c] = self.canvas.create_rectangle(
                    c * cell_size, r * cell_size,
                    (c + 1) * cell_size, (r + 1) * cell_size,
                    fill="white", outline="gray"
                )

        # Board data initialization
        self.board = np.zeros((rows, cols), dtype=int)

        # Set initial pattern and draw
        self.set_pattern("Glider")

        # Bind for editing
        self.canvas.bind("<Button-1>", self.toggle_cell)

        # Start update loop
        self.update()

    def set_pattern(self, pattern_name):
        """Clear the board and draw the selected pattern centred on the grid."""
        self.is_running = False
        self.update_controls()

        self.board.fill(0)

        pattern = patterns.get(pattern_name, [])
        if not pattern:
            self.draw_board()
            return

        # Determine the pattern’s bounding box
        r_vals = [r for r, _ in pattern]
        c_vals = [c for _, c in pattern]
        min_r, max_r = min(r_vals), max(r_vals)
        min_c, max_c = min(c_vals), max(c_vals)

        pattern_height = max_r - min_r + 1
        pattern_width = max_c - min_c + 1

        # Calculate the starting top-left corner to center the pattern
        start_r = (rows - pattern_height) // 2
        start_c = (cols - pattern_width) // 2

        # Apply the offsets, taking the pattern's internal origin into account
        for r, c in pattern:
            new_r = start_r + (r - min_r)
            new_c = start_c + (c - min_c)
            if 0 <= new_r < rows and 0 <= new_c < cols:
                self.board[new_r, new_c] = 1

        self.draw_board()

    def draw_board(self):
        for r in range(rows):
            for c in range(cols):
                color = "black" if self.board[r, c] else "white"
                self.canvas.itemconfig(self.cells[r, c], fill=color)

    def toggle_cell(self, event):
        c = event.x // cell_size
        r = event.y // cell_size
        if 0 <= r < rows and 0 <= c < cols:
            self.board[r, c] = 1 - self.board[r, c]
            color = "black" if self.board[r, c] else "white"
            self.canvas.itemconfig(self.cells[r, c], fill=color)

    def count_neighbors(self, r, c):
        total = 0
        for i in range(r - 1, r + 2):
            for j in range(c - 1, c + 2):
                if (i == r and j == c):
                    continue
                if 0 <= i < rows and 0 <= j < cols:
                    total += self.board[i, j]
        return total

    def toggle_run(self):
        self.is_running = not self.is_running
        self.update_controls()
        if self.is_running:
            self.update()

    def update_controls(self):
        if self.is_running:
            self.play_pause_btn.config(text="Pause")
        else:
            self.play_pause_btn.config(text="Play")

    def update(self):
        if self.is_running:
            self.next_generation()
            self.master.after(500, self.update)

    def next_step(self):
        if not self.is_running:
            self.next_generation()

    def next_generation(self):
        new_board = np.copy(self.board)
        changed_cells = []
        for r in range(rows):
            for c in range(cols):
                neighbors = self.count_neighbors(r, c)
                if self.board[r, c] == 1:
                    if neighbors < 2 or neighbors > 3:
                        new_board[r, c] = 0
                        changed_cells.append((r, c))
                else:
                    if neighbors == 3:
                        new_board[r, c] = 1
                        changed_cells.append((r, c))

        self.board = new_board
        for r, c in changed_cells:
            color = "black" if self.board[r, c] else "white"
            self.canvas.itemconfig(self.cells[r, c], fill=color)

    def clear_board(self):
        self.is_running = False
        self.update_controls()
        self.board.fill(0)
        self.draw_board()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Game of Life")
    game = GameOfLife(root)
    root.mainloop()