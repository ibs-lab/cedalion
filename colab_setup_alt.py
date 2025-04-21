#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Cedalion installation options.")

parser.add_argument(
    "--branch",
    default="dev",
    help="Branch of Cedalion to install (default: %(default)s)"
)
args = parser.parse_args()

print("Installing dependencies...")
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

subprocess.run(["pip", "install"] + packages, check=True)

print(f"Installing cedalion ({args.branch} branch)...")
cedalion_repo = f"git+https://github.com/ibs-lab/cedalion.git@{args.branch}"
subprocess.run(["pip", "install", cedalion_repo], check=True)

# Uninstall opt-einsum
subprocess.run(["pip", "uninstall", "-y", "opt-einsum"], check=True)

# Start PyVista in headless mode for Colab
try:
    import pyvista
    pyvista.start_xvfb()
    print("PyVista XVFB headless rendering enabled.")
except ImportError:
    print("WARNING: PyVista not installed yet, could not start XVFB.")

print("Setup complete.")
