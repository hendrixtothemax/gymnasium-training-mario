# Ensure conda is available
eval "$(conda shell.bash hook)"

# Create and activate environment
conda create --yes --name gymnasium-mario-env python=3.10
conda activate gymnasium-mario-env

# Install Conda-Forge Based Dependencies
conda install -c conda-forge libstdcxx-ng

# Install pip dependencies
pip install swig
pip install gymnasium[all] pillow pyboy keyboard
pip install stable_baselines3[extra]