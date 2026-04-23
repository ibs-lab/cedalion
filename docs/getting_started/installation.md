# Installation

Installing the Cedalion toolbox on your local machine takes 6 steps, which are summarized
below and then descibed in more detail in the following section.

## Summary

1. **Install conda** – Install either Miniconda or Anaconda for environment and package 
   management.
2. **Get the source code** – Clone the Cedalion GitHub repository. 
3. **Select a version** - Use the `main` branch for stable releases or `dev` for the most recent development version.
4. **Create a conda environment** – Create a new conda environment using the provided `environment_dev.yml` file.
5. **Install Cedalion** -- Install Cedalion in editable mode so you can use or modify it directly.
6. **Optional: Install NirFaster** - Install a supplemental package for simulating light
   propagation in tissue.

## Step-by-Step Instructions

### 1. Install conda

Cedalion depends on many third-party python libraries. To uniformly provide environments containing these dependencies across different platforms (Linux, Windows, MacOS) we curently rely on the [conda](https://docs.anaconda.com/working-with-conda/packages/install-packages/) package manager and the [conda-forge](https://conda-forge.org/docs/) package repository.

Install either **Miniconda** or **Anaconda**:

- **Miniconda**: A minimal Conda distribution. (Recommended)
  - [Miniconda Installation Guide](https://docs.anaconda.com/miniconda/install/)
- **Anaconda**: A larger distribution with many scientific Python packages included.
  - [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)


### 2. Get the source code

The source code of the Cedalion toolbox is hosted on our public 
[git repository](https://github.com/ibs-lab/cedalion). By following this guide, you will clone the public repository to your local machine. This clone contains the complete developement history and maintains a connection to the public repository. This way you can switch between version, pull updates and contribute changes back.

Open a terminal, navigate to the directory where you want the local repository to be stored, and run the clone command. This will create a new folder named `cedalion` in that location. 

(The `$` sign represents the terminal prompt and should not be copied. The `#` signs 
indicate comments. Placeholders are written as `<placeholder>` and should be replaced by 
the user.)

```bash
$ cd /path/to/install_directory

# for read-write access with a GitHub account:
$ git clone git@github.com:ibs-lab/cedalion.git

# alternatively for read-only access without a GitHub account:
$ git clone https://github.com/ibs-lab/cedalion.git
```

Alternatively, if you prefer using the **GitHub Desktop** application, please refer to its [documentation on cloning repositories](https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop).


### 3. Select a version

Select the desired version by choosing the appropriate branch in your local repository.

- The `main` branch serves as the release branch. Its latest commit always represents
  the most recent stable release. As the default branch, `main` is checked out
  when the repository is cloned.
- Older releases on `main` are marked with tags such as `v25.0.0`. 
- The `dev` branch contains the latest development version.

For a complete release history and details on differences between `dev` and `main`, see the [CHANGELOG](../CHANGELOG.md).

```bash
$ cd /path/to/install_directory/cedalion

# switch between branches
$ git switch <branchname>

# select specific versions
$ git switch -d <tagname>
```

Alternatively, if you prefer using the **GitHub Desktop** application, please refer to its [documentation on switching between branches](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop#switching-between-branches).


### 4. Create a conda environment

Next, create a [conda environment](https://docs.conda.io/projects/conda/en/stable/user-guide/concepts/environments.html) with all required dependendencies:

```bash
$ conda env create -n <environment_name> -f environment_dev.yml
```

Choose a descriptive name for the environment. Since you may need multiple environments over time (e.g., for projects using different Cedalion versions), a naming convention like `cedalion_YYMMDD` can be helpful, but feel free to use any scheme that works for you.


### 5. Install Cedalion 

Activate the new environment and install Cedalion in editable mode so you can 
use it and easily modify its functionality if needed.

```bash
$ conda activate <environment_name>
$ pip install -e . --no-deps
```

### 6. Optional: Install nirfaster-uFF

Cedalion supports two photon propagators to simulate light transport in tissue: [pmcx](https://mcx.space/) and [NIRFASTER-uFF](https://github.com/milabuob/nirfaster-FF). pmcx
is installed by default but it requires a GPU with CUDA support. NIRFASTER-uFF runs also 
on the CPU and can be installed by running:

```bash
$ bash install_nirfaster.sh CPU # or GPU
```

## Using the environment

To use your Cedalion installation, first activate the environment and then execute
python programs using Cedalion within that environment. 

For example, to run jupyter notebook:

```bash
$ conda activate <environment_name>
$ jupyter notebook
```

To test your installation, you can use the example notebook [examples/00_test_installation.ipynb](../examples/getting_started_io/00_test_installation.ipynb).

Alternatively, integrated development environments (IDEs) like **VSCode** have builtin 
support for [managing environments](https://code.visualstudio.com/docs/python/environments) and 
[working with Jupyter notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks). Make sure that you select the correct environment and Jupyter kernel.

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
$ pip install -e . --no-deps
```

Switching between different cedalion versions is then possible by activating the 
corresponding environment.


### During development

Cedalion's development happens in the dev branch. The cloned git repository contains the
complete development history and maintains the connection to our main repository at
GitHub. By pulling the recent changes from there or by checking out a commit from the past
the cedalion directory can be brought to any desired version. The conda environment
will then use the checked out version. 

Keep in mind that cedalion's dependencies changed over time. When pulling recent
changes from dev you might need to update or recreate the environment.


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
