@echo off
REM Rebuild the container
docker build -t maze-solver .

REM Run the container with volume mounts for input and output, and port mappings for TensorBoard and Dash
docker run -d --name maze-container -v "%cd%/host-input:/app/input" -v "%cd%/host-output:/app/output" -p 6006:6006 -p 8050:8050 maze-solver

REM Display container ID for easy access.
docker ps

echo Container started.  TensorBoard: http://localhost:6006/, Dash Dashboard: http://localhost:8050/