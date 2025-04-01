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

# Create a .pth file for the Colab environment
echo "Creating .pth file to add site-packages to PYTHONPATH..."
echo "$SITE_PACKAGES_PATH" > /usr/local/lib/python3.11/dist-packages/venv_site_packages.pth

# Create a helper script to activate the environment and install any missing packages
cat > /content/activate_venv.py << EOF
import sys
import site
import os

# Add the site-packages to path
site_packages = "$SITE_PACKAGES_PATH"
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)
    print(f"Added {site_packages} to sys.path")

# Try importing cedalion to verify
try:
    import cedalion
    print(f"Successfully imported cedalion from {cedalion.__file__}")
except ImportError as e:
    print(f"Error importing cedalion: {e}")
    print("Current sys.path:", sys.path)
EOF

echo "Setup complete. To ensure cedalion is in your path, run the following line in your notebook:"
echo "exec(open('/content/activate_venv.py').read())"