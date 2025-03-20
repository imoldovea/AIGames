FROM python:3.12

##Alternative
#FROM ubuntu:latest
#RUN apt-get update && apt-get install -y python3 python3-pip


# Update package lists and upgrade existing packages
RUN apt-get update && apt-get upgrade -y


# Install graphics dependencies
RUN apt-get install -y libgl1-mesa-glx

# Install other dependencies (if any)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Command to run your application
CMD ["python3", "rnn2_maze_solver.py"]