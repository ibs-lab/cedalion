# cedalion - fNIRS analysis toolbox

To avoid misinterpretations and to facilitate studies in naturalistic environments, fNIRS measurements will increasingly be combined with recordings from physiological sensors and other neuroimaging modalities.
The aim of this toolbox is to facilitate this kind of analyses, i.e. it should allow the easy integration of machine learning techniques and provide unsupervised decomposition techniques for
multimodal fNIRS signals.

## Documentation

The [documentation](https://doc.ml.tu-berlin.de/cedalion/docs) contains
[installation instructions](https://doc.ml.tu-berlin.de/cedalion/docs/getting_started/installation.html) as
well as several [example notebooks](https://doc.ml.tu-berlin.de/cedalion/docs/examples/index.html)
that illustrate the functionality of the toolbox.

## Development environment

To create a conda environment with the necessary dependencies run:

```
$ conda env create -n cedalion -f environment_dev.yml
```

Afterwards activate the environment and add an editable install of `cedalion` to it:
```
$ conda activate cedalion
$ pip install -e .
$ bash install_nirfaster.sh CPU # or GPU
```

This will also install Jupyter Notebook to run the example notebooks.

If conda is too slow consider using the faster drop-in replacement [mamba](https://mamba.readthedocs.io/en/latest/).
If you have Miniconda or Anaconda you can install mamba with:
'''
$ conda install mamba -c conda-forge
'''
and then create the environment with
```
$ mamba env create -n cedalion -f environment_dev.yml
```
Please note: If this does not socceed there is another route to go:
Install the libmamba solver
'''
$ conda install -n base conda-libmamba-solver
'''
and then build the environment with the --solver=libmamba
```
$ conda env create -n cedalion -f environment_dev.yml --solver=libmamba
```
