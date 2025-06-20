#!/usr/bin/env python3
import os
import argparse
import subprocess
from google.colab import drive

def kill_current_runtime():
    os.kill(os.getpid(), 9)

# mount google drive
drive.mount("/content/drive", force_remount=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Cedalion installation options.")

parser.add_argument(
    "--branch",
    default="dev",
    help="Branch of Cedalion to install (default: %(default)s)",
)
args = parser.parse_args()

try:
    import cedalion
    CEDALION_INSTALLED = True
except ImportError:
    CEDALION_INSTALLED = False

if not CEDALION_INSTALLED:
    cedalion_repo = f"git+https://github.com/ibs-lab/cedalion.git@{args.branch}"

    print("Installing dependencies...")
    subprocess.run(["uv", "pip", "install", cedalion_repo], check=True)
    subprocess.run(["uv", "pip", "uninstall", "opt-einsum"], check=True)

    print("Dependencies installed. Killing this runtime. Please rerun the notebook.")
    kill_current_runtime()

else:
    # Start PyVista in headless mode for Colab
    print("Setting Pyvista options...")
    try:
        import pyvista as pv

        pv.global_theme.jupyter_backend = "static"
        pv.global_theme.notebook = True
        pv.start_xvfb()
        print("Setup complete. Ready to proceed with the notebook.")
    except ImportError:
        print("WARNING: PyVista not installed yet, could not start XVFB.")
