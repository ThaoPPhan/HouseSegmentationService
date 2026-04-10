# Use a stable, slim version of Python
FROM python:3.11-slim

# Fix for Exit Code 100: Tell apt-get we are in a non-interactive shell
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The rest of your Dockerfile...
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]