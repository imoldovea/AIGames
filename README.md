# AIGames

This is an AI learning Python Project. It uses synthetic generated maze data. The project starts with Classical algorithms as backtracking and BFS. BFS is solutions are used in training RNNS. 

RNN implementation is  GRU,LSTM
	- Input parameters are: local context (state of N, E, S, W cells around the current position), position relative to the starting point, and number of steps in in the solution
	- Outputs are the 4th direction of travel on the maze and 1 exit predicting if the next step of the solution is at the exit point

The concept is to implement a solution working form the viewpoint of inside the maze, as it is the case with backtracking. The solver algorithm will not have an overview of the entire maze, unless is has explored those parts. 

Next steps: 
	- Implement a large language model solution
Implement a genetic algorithm solution.  