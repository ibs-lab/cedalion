# Installation

Get the latest release of the cedalion toolbox from our public 
[github repository](https://github.com/ibs-lab/cedalion). Releases can be found in
the `main` branch of the repository wheras development happens in the `dev` branch.

Cedalion depends on many third-party python libraries. To uniformly provide environments 
containing these dependencies across different platforms (Linux, Windows, MacOS) we rely 
on the [conda](https://docs.anaconda.com/working-with-conda/packages/install-packages/) 
package manager and the [conda-forge](https://conda-forge.org/docs/) package
repository.

## Running Notebooks in Google Colab

If you would like to test cedalion before going through the installation process on your local machine, you can run the example notebooks through Google Colab. This requires a Google account with approximately 50MB available storage space. To open a notebook using Google Colab, follow the steps below. The first time you open a notebook in Colab, setup can take 10-20 minutes. For subsequent notebooks the steps remain the same, but setup should be much faster.

Click the link at the top of an example notebook to open it in Google Colab. Then, run the first cell to install the required dependencies in a virtual environment on your Google Drive. Follow the prompts to sign into your Google Drive account. At the end of the setup process you may also be prompted to run the cell a second time. Then you can proceed with the rest of the example notebook.


## Development

Follow the installation instructions to install the [Miniconda](https://docs.anaconda.com/miniconda/install/) or [Anaconda](https://docs.anaconda.com/anaconda/install/) distribution.

Clone the git repository to a directory on your machine:

```
$ git clone git@github.com:ibs-lab/cedalion.git
```

This creates a new directory `cedalion` in your current folder. It checks out the latest 
stable release from the `main` branch. If you intend to contribute to cedalion please 
check out the `dev` branch.

To create a conda environment named `cedalion` with the necessary dependendencies enter 
this checked-out directory and run:

```
$ conda env create -n cedalion -f environment_dev.yml
```

Afterwards activate the environment and add an editable install of `cedalion` to it:
```
$ conda activate cedalion
$ pip install -e .
```

This will also install Jupyter Notebook to run the example notebooks.


```{admonition} If it's slow...
:class: tip

To create the environment conda needs to find a set of packages that fulfills all
requirements. If conda needs too much time to find a solution, there are two ways to
speed it up.

1. Make sure that you are using a recent version of conda (> 23.10) that uses 
[libmamba-solver](https://conda.github.io/conda-libmamba-solver/user-guide/) to resolve dependcies. (***recommended***)

2. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), 
a drop-in replacement for conda.
```

## Development using Hatch

Alternatively, there is preliminary support for using the Python project manager [Hatch](https://hatch.pypa.io/latest/). Hatch helps with the handling of the python environments and
offers a simple interface to perform common tasks.

As a tool to manage cedalion's environment, hatch and its dependencies must be [installed](https://hatch.pypa.io/1.13/install/) in a separate environment, like for example the `base` environment of a Miniconda/Anaconda installation or using [pipx](https://pipx.pypa.io/latest/):

```
$ pipx install hatch
$ pipx inject hatch hatch-vcs hatch-conda hatchling
```

Then clone cedalion's git repository and change to the checked-out directory:

```
$ git clone git@github.com:ibs-lab/cedalion.git
$ cd cedalion
```

To create the environment run and install `cedalion` in editable mode run:
```
$ hatch env create
```

To run the tests call:
```
$ hatch test
```

To locally build the documenation run:
```
$ hatch run build_docs
```

## Production

The same procedure as above applies. However, make sure to use a released version
from the main branch.


## Container Environments

### Apptainer

For running cedalion in a headless environment we provide an [Apptainer](https://apptainer.org/)
definition file. This container provides a populated conda environment 
and runs the X Window virtual Framebuffer (Xvfb) needed for the 3D plotting functionality 
of pyvista. The directory with the cedalion source code is expected to be mounted under 
`/app`. It is added as an editable install to the conda environment, i.e. changes
made in the host directory propagate into the container.

XVfb needs write access under `/var/lib/xkb` which is not available from inside the
container. As workaround we bind a writable directory from the host to this folder. Probably a more elegant solution exists.

#### Building the container:

Clone the cedalion repository to `./cedalion`
```
$ git clone https://github.com/ibs-lab/cedalion.git
```

Build the container. The cedalion source code needs to be mounted under `/app`.
```
$ apptainer build --bind `pwd`/cedalion:/app cedalion.sif cedalion/cedalion.def
```

#### Run jupyter notebook in the container

```
$ mkdir -p xkb
$ apptainer run --nv --bind `pwd`/xkb:/var/lib/xkb,`pwd`/cedalion:/app cedalion.sif jupyter notebook --ip 0.0.0.0 --no-browser
```

### Docker

- WIP: see [Nils' branch](https://github.com/ibs-lab/cedalion/tree/docker)
