# cedalion - fNIRS analysis toolbox

To avoid misinterpretations and to facilitate studies in naturalistic environments, fNIRS measurements will increasingly be combined with recordings from physiological sensors and other neuroimaging modalities.
The aim of this toolbox is to facilitate this kind of analyses, i.e. it should allow the easy integration of machine learning techniques and provide unsupervised decomposition techniques for
multimodal fNIRS signals.


## Development environment

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

If conda is too slow consider using the faster drop-in replacement [mamba](https://mamba.readthedocs.io/en/latest/).
