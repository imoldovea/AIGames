# Project Guidelines

This is an AI learning Python project. It uses synthetically generated maze data. The project starts with classical
algorithms such as backtracking and BFS. BFS is solutions are used in training RNNs.

RNN implementation is GRU,LSTM
- Input parameters are: local context (state of N, E, S, W cells around the current position), position relative to the
starting point, and number of steps in in the solution
- Outputs are the 4th direction of travel on the maze and 1 exit predicting if the next step of the solution is at the
exit point

The concept is to implement a solution working from the viewpoint of inside the maze, as is the case with backtracking.
The solver algorithm will not have an overview of the entire maze unless it has explored those parts.

Next steps:
- Implement a large language model solution

- Implement a genetic algorithm solution.