# gameoflife_gui.py

import tkinter as tk

import numpy as np

rows, cols = 20, 20
cell_size = 20

patterns = {
    # Patterns remain the same
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
    def __init__(self, master):
        self.master = master
        self.is_running = False

        # Pattern buttons
        pattern_frame = tk.Frame(master)
        pattern_frame.pack()
        for pattern_name in patterns:
            btn = tk.Button(pattern_frame, text=pattern_name,
                            command=lambda name=pattern_name: self.set_pattern(name))
            btn.pack(side=tk.LEFT)

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

        # Board initialization
        self.board = np.zeros((rows, cols), dtype=int)
        self.set_pattern("Glider")
        self.draw_board()

        # Bind for editing
        self.canvas.bind("<Button-1>", self.toggle_cell)

        # Start update loop
        self.update()

    def set_pattern(self, pattern_name):
        self.board.fill(0)
        for (r, c) in patterns.get(pattern_name, []):
            if 0 <= r < rows and 0 <= c < cols:
                self.board[r, c] = 1
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(rows):
            for c in range(cols):
                color = "black" if self.board[r, c] else "white"
                self.canvas.create_rectangle(
                    c * cell_size, r * cell_size,
                    (c + 1) * cell_size, (r + 1) * cell_size,
                    fill=color, outline="gray"
                )

    def toggle_cell(self, event):
        c = event.x // cell_size
        r = event.y // cell_size
        if 0 <= r < rows and 0 <= c < cols:
            self.board[r, c] = 1 - self.board[r, c]
            self.draw_board()

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

    def clear_board(self):
        self.board.fill(0)
        self.draw_board()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Game of Life")
    game = GameOfLife(root)
    root.mainloop()