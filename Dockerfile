# Ensure you are using a slim Debian-based image
FROM python:3.10-slim

# Fix for Exit Code 100: Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Update and install with extra flags to ignore transient mirror issues
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*