# Use a lightweight official Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /code

# Copy and install dependencies first (Docker layer caching — faster rebuilds)
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the project into the container
COPY . /code

# Create logs directory so FileHandler doesn't crash (bulletproof practice)
RUN mkdir -p /code/logs

# Render uses $PORT (default 10000).
# Using the shell form of CMD so $PORT is expanded at runtime.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}