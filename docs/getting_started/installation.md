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
