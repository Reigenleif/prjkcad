FROM condaforge/miniforge3:latest

# Set working directory
WORKDIR /app

# Set non-interactive and environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies needed for OpenCASCADE and graphics rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCASCADE (pythonocc-core) and Python 3.10 via conda-forge
RUN conda install -y -c conda-forge \
    python=3.10 \
    pythonocc-core=7.7.2 \
    && conda clean -afy

# Install PyTorch CPU and Python dependencies for FastAPI backend
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    transformers \
    trimesh \
    scipy \
    numpy \
    python-multipart \
    h5py \
    tqdm \
    Pillow

# Copy core modules and backend code into the container
COPY utils /app/utils
COPY gui/backend /app/gui/backend
COPY gui/model_runner.py /app/gui/model_runner.py
COPY gui/assets /app/gui/assets
COPY gui/__init__.py /app/gui/__init__.py

# Ensure renders output directory exists
RUN mkdir -p /app/gui/renders

# Expose backend port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI backend server
CMD ["uvicorn", "gui.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
