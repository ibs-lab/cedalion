#!/bin/bash -e

if [ "$#" -ne 1 ]; then
    echo "usage: ${0} <path to docs>"
    exit 1
fi

path_to_docs=$1

echo "Prefetch datasets"
python scripts/prefetch_docs_datasets.py

cd ${path_to_docs}

echo "Building example notebook"
cd examples
make -j 2 notebooks

echo "Building html documentation"
cd ..
make clean

sphinx-apidoc -f -o api ../src/cedalion

make html

echo "Building tarball"
cd _build/html
tar cvzf ../html.tar.gz *
