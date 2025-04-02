#!/usr/bin/env python3
import os
import sys
import subprocess

# Define paths for the virtual environment and its site-packages directory.
VENV_PATH = "/content/drive/MyDrive/vir_env"
PYTHON_BIN = os.path.join(VENV_PATH, "bin", "python")
SITE_PACKAGES_PATH = os.path.join(VENV_PATH, "lib", "python3.11", "site-packages")

print(f"Using virtual environment path: {VENV_PATH}")
print("Current working directory:", os.getcwd())

# Ensure we're in /content, which is the expected working directory in Colab.
if os.getcwd() != "/content":
    print("WARNING: Not in expected directory. Changing directory to /content.")
    os.chdir("/content")

# If the virtual environment does not exist, create it and install dependencies.
if not os.path.isdir(VENV_PATH):
    print("Virtual environment not found. Creating one...")
    subprocess.run(["pip", "install", "virtualenv"], check=True)
    subprocess.run(["virtualenv", VENV_PATH], check=True)

    # Confirm that the environment was created correctly.
    if not os.path.isfile(PYTHON_BIN):
        print(f"ERROR: Virtual environment not created correctly at {VENV_PATH}")
        sys.exit(1)

    print(f"Installing dependencies using {PYTHON_BIN}...")
    # List packages to install. Adjust versions and packages as necessary.
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

    # Install cedalion directly from the git repository.
    subprocess.run([PYTHON_BIN, "-m", "pip", "install", "git+https://github.com/ibs-lab/cedalion.git"], check=True)

    print("Setup complete.")
else:
    print(f"Virtual environment already exists at {VENV_PATH}.")

# Update the notebook’s sys.path and PYTHONPATH so that the virtual environment packages take priority.
if SITE_PACKAGES_PATH not in sys.path:
    sys.path.insert(0, SITE_PACKAGES_PATH)
os.environ["PYTHONPATH"] = SITE_PACKAGES_PATH + ":" + os.environ.get("PYTHONPATH", "")

# Force installation of the desired numpy version in the current process.
subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.0"], check=True)

# Verify that the correct numpy version is in use.
import numpy
print("Numpy version:", numpy.__version__)
