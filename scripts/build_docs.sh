#!/bin/bash -e

if [ "$#" -ne 1 ]; then
    echo "usage: ${0} <path to docs>"
    exit 1
fi

path_to_docs=$1

cd ${path_to_docs}

echo "Building example notebook"
cd examples
make notebooks

echo "Building html documentation"
cd ..
make clean

sphinx-apidoc -o api ../src/cedalion

make html

echo "Building tarball"
cd _build/html
tar cvzf ../html.tar.gz *
