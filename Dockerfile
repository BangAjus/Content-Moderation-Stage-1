# Use official Miniconda base layer
FROM continuumio/miniconda3:latest

# Force system shell to handle bash executions natively
SHELL ["/bin/bash", "-c"]

# Set deterministic execution and framework path targets
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

# 1. Install core system video/audio decoding layers inside Linux
# (Crucial fix: Without these, cv2 and FFmpeg subprocess pipes will crash)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Establish application directory inside container space
WORKDIR /app

# 2. Bind Anaconda versioning schema
COPY environment.yml .

# 3. Compile Conda environment and clean cache targets to save space
RUN conda env create -f environment.yml && \
    conda clean -afy

# Update the environment path variables permanently
ENV PATH /opt/conda/envs/stage1_env/bin:$PATH

# 4. Copy the isolated model downloader file 
# (Leverages Docker caching layers so code adjustments won't trigger re-downloads)
COPY download_models.py .
RUN python download_models.py

# 5. Pack your updated production scripts and pipeline structures
COPY . .

# Set default application runtime variables
ENV CURRENT_DEVICE=cpu

# Tell Python to look at the /app root for all module paths
ENV PYTHONPATH=/app

# Default command target to kickstart processing (Adjust to your primary script execution target)
CMD ["python", "model/model_structure.py"]