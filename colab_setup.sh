#!/bin/bash
set -e  # Exit immediately if a command exits with non-zero status

# Path settings
VENV_PATH="/content/drive/MyDrive/vir_env"
PYTHON_BIN="$VENV_PATH/bin/python"
SITE_PACKAGES_PATH="$VENV_PATH/lib/python3.11/site-packages"

echo "Using virtual environment path: $VENV_PATH"
echo "Current working directory: $(pwd)"

# Check if we're in the right working directory
if [ "$(pwd)" != "/content" ]; then
  echo "WARNING: Not in expected directory. Moving to /content"
  cd /content
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found. Creating one..."
    pip install virtualenv
    
    # Create the directory structure first to ensure correct location
    mkdir -p "$VENV_PATH"
    
    # Use absolute path for virtualenv
    virtualenv "$VENV_PATH"
    
    # Verify the environment was created correctly
    if [ ! -f "$PYTHON_BIN" ]; then
        echo "ERROR: Virtual environment not created correctly at $VENV_PATH"
        exit 1
    fi

    # Install required dependencies
    echo "Installing dependencies using $PYTHON_BIN"
    "$PYTHON_BIN" -m pip install click==8.1 h5py==3.11 ipython==8.13.2 ipywidgets==8.1.2 jupyter \
    jupyter_client==7.4.9 matplotlib==3.9 nibabel==5.2 nilearn==0.10 notebook==6.5.4 \
    numpy==1.26 opencv-python pandas==2.2 pint-xarray==0.3 pip pooch==1.8 \
    pybids==0.16 pytest pytest-cov pywavelets==1.6 ruff scikit-image==0.24 \
    scikit-learn==1.5 scipy==1.14 seaborn==0.13 statsmodels==0.14 strenum==0.4 \
    xarray==2024.6 trimesh==4.4 pyvista==0.44 trame==3.6 trame-vtk==2.8 \
    trame-vuetify==2.6 trame-components==2.3 vtk==9.2.6 ipympl mne==1.7 \
    mne-bids==0.15 mne-nirs==0.6 pywavefront==1.3 setuptools-scm snirf==0.8 \
    pmcx==0.3.3 pmcxcl==0.2.0
    
    # Install cedalion
    "$PYTHON_BIN" -m pip install git+https://github.com/ibs-lab/cedalion.git
    echo "Setup complete."
else
    echo "Virtual environment already exists at $VENV_PATH."
fi

# Create a more radical approach - using symlinks to override system packages
echo "Setting up package overrides..."

# For numpy specifically, temporarily move the original and create a symlink to our version
if [ -d "/usr/local/lib/python3.11/dist-packages/numpy" ] && [ -d "$SITE_PACKAGES_PATH/numpy" ]; then
    echo "Redirecting numpy to use virtual environment version..."
    
    # Backup the original numpy if not already backed up
    if [ ! -d "/usr/local/lib/python3.11/dist-packages/numpy_original" ]; then
        mv /usr/local/lib/python3.11/dist-packages/numpy /usr/local/lib/python3.11/dist-packages/numpy_original
    fi
    
    # Create symlink from system location to our venv version
    ln -sf "$SITE_PACKAGES_PATH/numpy" /usr/local/lib/python3.11/dist-packages/numpy
    
    echo "Numpy has been redirected to virtual environment version."
else
    echo "WARNING: Either system numpy or virtual environment numpy not found."
    echo "System numpy: $([[ -d /usr/local/lib/python3.11/dist-packages/numpy ]] && echo "Found" || echo "Not found")"
    echo "Venv numpy: $([[ -d $SITE_PACKAGES_PATH/numpy ]] && echo "Found" || echo "Not found")"
fi

# Create a startup script to ensure proper package overrides
cat > /content/ensure_venv_packages.py << EOF
import sys
import os
import importlib.util

def ensure_venv_packages():
    venv_path = "$SITE_PACKAGES_PATH"
    
    # Add venv path to beginning of sys.path
    if venv_path not in sys.path:
        sys.path.insert(0, venv_path)
        print(f"Added {venv_path} to sys.path")
    else:
        # Make sure it's at the beginning
        sys.path.remove(venv_path)
        sys.path.insert(0, venv_path)
    
    # Verify numpy version
    try:
        import numpy
        print(f"Using numpy from: {numpy.__file__}")
        print(f"Numpy version: {numpy.__version__}")
    except ImportError as e:
        print(f"Error importing numpy: {e}")

if __name__ == "__main__":
    ensure_venv_packages()
EOF

# Create IPython startup file to run our script
mkdir -p /root/.ipython/profile_default/startup/
cat > /root/.ipython/profile_default/startup/01-venv-packages.py << EOF
# This script runs automatically when IPython starts
import sys
import os

try:
    exec(open('/content/ensure_venv_packages.py').read())
except Exception as e:
    print(f"Error ensuring venv packages: {e}")
EOF

# Verify the setup
echo "Verifying the environment setup..."
python /content/ensure_venv_packages.py

echo "Setup complete. Your virtual environment's packages should now override Colab's defaults."
echo "To verify in your notebook, run:"
echo "import numpy; print(numpy.__file__, numpy.__version__)"