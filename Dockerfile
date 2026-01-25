# Base image
FROM ubuntu:22.04

#docker build -t adl_conda .
#docker run -dit --name adl_hw1 --rm -u $(id -u):$(id -g) -v /Users/jjonnala/tmp/courses/adl/git/adl/hw/hw1:/app  -w /app adl_conda bash
# docker exec -it -u jjonnala adl_hw1 bash

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install build tools and Python build dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download Miniconda (latest)
RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh

# Optional: make sure conda itself is up to date
#RUN conda update -n base -c defaults conda -y

# Create a non-root user
RUN useradd -m jjonnala

# Set working directory
WORKDIR /app

# Switch to non-root user
USER jjonnala 

CMD ["/bin/bash"]
