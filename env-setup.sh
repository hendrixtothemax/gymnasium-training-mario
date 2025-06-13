# Ensure conda is available
eval "$(conda shell.bash hook)"

# Create and activate environment
conda create --yes --name gymnasium-mario-env python=3.11
conda activate gymnasium-mario-env

# Install dependencies
pip install pillow pyboy