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

# Create a .pth file for the Colab environment to ensure it's always on the path
echo "Creating .pth file to add site-packages to PYTHONPATH..."
echo "$SITE_PACKAGES_PATH" > /usr/local/lib/python3.11/dist-packages/venv_site_packages.pth

# Create a Colab extension that modifies the path at runtime
mkdir -p /usr/local/lib/python3.11/dist-packages/google/colab/

cat > /usr/local/lib/python3.11/dist-packages/google/colab/venv_path_modifier.py << EOF
import os
import sys
import site
import importlib.util

# Add hook that runs on startup
def _modify_path():
    venv_site_packages = "$SITE_PACKAGES_PATH"
    
    # Remove any existing instances of venv_site_packages from sys.path
    sys.path = [p for p in sys.path if p != venv_site_packages]
    
    # Insert at the beginning to prioritize these packages
    sys.path.insert(0, venv_site_packages)
    
    # Force numpy to be imported from venv if present
    numpy_path = os.path.join(venv_site_packages, "numpy")
    if os.path.exists(numpy_path):
        # Remove numpy from modules if already imported
        if "numpy" in sys.modules:
            del sys.modules["numpy"]
        
        # Ensure numpy is imported from venv
        try:
            spec = importlib.util.spec_from_file_location(
                "numpy", 
                os.path.join(numpy_path, "__init__.py")
            )
            if spec:
                numpy_module = importlib.util.module_from_spec(spec)
                sys.modules["numpy"] = numpy_module
                spec.loader.exec_module(numpy_module)
                print(f"Using numpy from {numpy_path}")
        except Exception as e:
            print(f"Warning: Could not force numpy import: {e}")

# Run the path modification
_modify_path()
EOF

# Create sitecustomize.py to ensure our path modifications happen automatically
cat > /usr/local/lib/python3.11/dist-packages/sitecustomize.py << EOF
import sys
import os

# Import our path modifier
try:
    from google.colab import venv_path_modifier
    print("Virtual environment path configured automatically")
except ImportError as e:
    print(f"Warning: Could not import path modifier: {e}")

# Print verification info
print(f"PYTHONPATH priority: {sys.path[:3]}")
EOF

# Make sure numpy works from the venv
echo "Verifying the environment setup..."
python -c "import sys; import numpy; print(f'Using numpy from: {numpy.__file__}'); print(f'Python path priority: {sys.path[:3]}')"

echo "Setup complete. Your virtual environment packages should now be automatically available in all notebooks."
echo "The virtual environment's numpy should now override Colab's default numpy."