# Use a lightweight official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy and install dependencies first (Docker layer caching — faster rebuilds)
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the project into the container
COPY . /code

# Create logs directory so FileHandler doesn't crash (bulletproof practice)
RUN mkdir -p /code/logs

# Hugging Face Spaces exposes port 7860 by default — must use this port
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]