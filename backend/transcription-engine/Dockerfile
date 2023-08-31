# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:11.7.1-base-ubuntu20.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=non-interactive

# Install system packages and Python 3.10, set python3.10 and pip3.10 as defaults
RUN apt-get update -y && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y \
        python3.10 \
        python3.10-distutils \
        ffmpeg \
        git \
        wget \
        tini \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf $(which pip) /usr/local/bin/pip3 \
    && ln -sf $(which pip) /usr/local/bin/pip3.10

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt

# Copy application files
COPY src/transcription-engine/* data/test-audio-128kbps.mp3 /app/

# Set tini as the entry point and run transcript.py
ENTRYPOINT ["/usr/bin/tini", "--", "python3", "/app/transcribe_test.py"]