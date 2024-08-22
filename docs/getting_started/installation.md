# Installation

## Development

To create a conda environment with the necessary dependendencies run:

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

1. Install [libmamba-solver](https://conda.github.io/conda-libmamba-solver/user-guide/)
   and configure conda to use it. (***recommended***)

2. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), a drop-in replacement for conda.

Additional guidance can be found in the the [environments section](environments.md).
```



## Production

- TBD
- currently, no fixed release. not on pypi.



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