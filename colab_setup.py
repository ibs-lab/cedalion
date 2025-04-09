#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Set up virtual environment for Cedalion on Google Colab.")
parser.add_argument(
    "--path",
    default="/content/drive/MyDrive/cedalion_files",
    help="Base path in Google Drive to install files (default: %(default)s)"
)
parser.add_argument(
    "--branch",
    default="dev",
    help="Branch of Cedalion to install (default: %(default)s)"
)
args = parser.parse_args()

# Define paths
VENV_PATH = os.path.join(args.path, f"venv_{args.branch}")
PYTHON_BIN = os.path.join(VENV_PATH, "bin", "python")
SITE_PACKAGES_PATH = os.path.join(VENV_PATH, "lib", "python3.11", "site-packages")

print(f"Using virtual environment path: {VENV_PATH}")
print(f"Installing Cedalion from branch: {args.branch}")
print("Current working directory:", os.getcwd())

# Ensure working directory is /content in Colab
if os.getcwd() != "/content":
    print("WARNING: Not in expected directory. Changing directory to /content.")
    os.chdir("/content")

# Create the virtual environment if it doesn't exist
if not os.path.isdir(VENV_PATH):
    print("Virtual environment not found. Creating one...")
    subprocess.run(["pip", "install", "virtualenv"], check=True)
    subprocess.run(["virtualenv", VENV_PATH], check=True)

    if not os.path.isfile(PYTHON_BIN):
        print(f"ERROR: Virtual environment not created correctly at {VENV_PATH}")
        sys.exit(1)

    print(f"Installing dependencies using {PYTHON_BIN}...")
    packages = (
        "click==8.1 h5py==3.11 ipython==8.13.2 ipywidgets==8.1.2 jupyter "
        "jupyter_client==7.4.9 matplotlib==3.9 nibabel==5.2 nilearn==0.10 notebook==6.5.4 "
        "numpy==1.26 opencv-python pandas==2.2 pint-xarray==0.3 pip pooch==1.8 "
        "pybids==0.16 pytest pytest-cov pywavelets==1.6 ruff scikit-image==0.24 "
        "scikit-learn==1.5 scipy==1.14 seaborn==0.13 statsmodels==0.14 strenum==0.4 "
        "xarray==2024.6 trimesh==4.4 pyvista==0.44 trame==3.6 trame-vtk==2.8 "
        "trame-vuetify==2.6 trame-components==2.3 vtk==9.2.6 ipympl mne==1.7 "
        "mne-bids==0.15 mne-nirs==0.6 pywavefront==1.3 setuptools-scm snirf==0.8 "
        "pmcx==0.3.3 pmcxcl==0.2.0"
    ).split()

    subprocess.run([PYTHON_BIN, "-m", "pip", "install"] + packages, check=True)

    cedalion_repo = f"git+https://github.com/ibs-lab/cedalion.git@{args.branch}"
    subprocess.run([PYTHON_BIN, "-m", "pip", "install", cedalion_repo], check=True)

    print("Setup complete.")
else:
    print(f"Virtual environment already exists at {VENV_PATH}.")

# Update sys.path and PYTHONPATH for the notebook
if SITE_PACKAGES_PATH not in sys.path:
    sys.path.insert(0, SITE_PACKAGES_PATH)
os.environ["PYTHONPATH"] = SITE_PACKAGES_PATH + ":" + os.environ.get("PYTHONPATH", "")

# Force reinstall numpy
subprocess.run(["pip", "uninstall", "-y", "numpy"], check=True)
subprocess.run(["pip", "install", "--force-reinstall", "numpy==1.26.0"], check=True)

# Check numpy version
import numpy
print("Numpy version:", numpy.__version__)
if numpy.__version__ != "1.26.0":
    print("WARNING: Numpy version is not 1.26.0. Please restart the runtime (Ctrl-M .) and re-run this cell.")
else:
    print("Numpy version is correct. Ready to proceed with the notebook.")