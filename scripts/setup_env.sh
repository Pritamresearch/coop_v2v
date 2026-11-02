#!/bin/bash
# Setup script for Kelvin2
# Run once: bash scripts/setup_env.sh

ENV_NAME=coop_v2v

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python-headless matplotlib pandas shapely pyyaml

# Optional but recommended (LiDAR PCD loading)
pip install open3d

echo "Environment $ENV_NAME is ready."
echo "Activate with:  conda activate $ENV_NAME"
