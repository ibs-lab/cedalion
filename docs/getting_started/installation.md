# Installation

Get the latest release of the cedalion toolbox from our public 
[github repository](https://github.com/ibs-lab/cedalion). Releases can be found in
the `main` branch of the repository wheras development happens in the `dev` branch.

Cedalion depends on many third-party python libraries. To uniformly provide environments 
containing these dependencies across different platforms (Linux, Windows, MacOS) we rely 
on the [conda](https://docs.anaconda.com/working-with-conda/packages/install-packages/) 
package manager and the [conda-forge](https://conda-forge.org/docs/) package
repository.

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

Select a descriptive name for the environment. Keep in mind that over time you
may want to have multiple environments in parallel, for example when you are working
on two projects that use different versions of cedalion. A naming scheme that
includes the current date ('cedalion_YYMMDD') is practical but you are free to choose
whatever works best for you.

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

## Updating

### Updating between releases

In the past, you cloned the git repository to a local directory using the last released 
version on the main branch. During installation, you created a conda environment and 
added cedalion from that directory to the environment.

Updating to a newer version is easiest done by cloning the git repository again to a 
different folder and creating a new environment. This way, the installed version remains 
usable. It also guarantees that the new environment contains any updated dependencies.

The following example uses the version suffix in the directory and  environment name.

```
$ git clone git@github.com:ibs-lab/cedalion.git path/to/cedalion_v25.1.0
$ cd path/to/cedalion_v25.1.0
$ conda env create -n cedalion_v25.1.0 -f environment_dev.yml
$ conda activate cedalion_v25.1.0
$ pip install -e .
```

Switching between the different cedalion versions is then possible by activating the 
corresponding environment.

### During development

Cedalion's development happens in the dev branch. The cloned git repository contains the
complete development history and maintains the connection to our main repository at
GitHub. By pulling the recent changes from there or by checking out a commit from the past
the cedalion directory can be brought to any desired version. The conda environment
will then use the checked out version. 

Keep in mind that the cedalion's dependencies changed over time. When pulling recent
changes from dev you might need to update or recreate the environment.



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
