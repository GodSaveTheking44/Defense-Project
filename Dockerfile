# Use the official Ultralytics image which has CUDA, PyTorch, OpenCV, and all requirements pre-installed
FROM ultralytics/ultralytics:latest

# Set working directory
WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for uvicorn, websockets, and sound/graphics fallbacks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi \
    uvicorn \
    scipy \
    pydantic

# Copy the rest of the application
COPY . .

# Expose port for the Web Dashboard
EXPOSE 8000

# Expose port for CoT UDP stream (multicast address)
EXPOSE 24910/udp

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: run in headless mode with web streaming and CoT active
ENTRYPOINT ["python", "cli.py"]
CMD ["--headless", "--web", "--cot"]
