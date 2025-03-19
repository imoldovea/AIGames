# Use an official Python image as the base
FROM python:3.12-slim


# Set the working directory in the container
WORKDIR /app

# Copy the list of dependencies into the container
COPY requirements.txt ./

# Create input and output directories if they don't exist
RUN mkdir -p ./input ./output

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Set the command to execute your main script.
# This will run the main function in rnn2_maze_solver.py
CMD ["python", "rnn2_maze_solver.py"]