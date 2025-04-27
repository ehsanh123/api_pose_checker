FROM python:3.10-slim

# Install system dependencies (GUI, audio, media support)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libportaudio2 \
    libasound2-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables to avoid permission issues
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make sure ffmpeg is linked for pydub
ENV PATH="/usr/bin/ffmpeg:$PATH"

# Expose port as expected by Hugging Face
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "7860"]