# Use slim and pin the version
FROM python:3.12.2-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory early
WORKDIR /app

# Create output directory
RUN mkdir -p /app/output


# Copy requirements first to benefit from caching
COPY requirements.txt .

# Use pip cache to speed up rebuilds (requires BuildKit)
#RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the source
COPY . .

# Expose necessary ports
EXPOSE 8050
EXPOSE 6006

# Run the main app
CMD ["python3", "rnn2_maze_solver.py"]