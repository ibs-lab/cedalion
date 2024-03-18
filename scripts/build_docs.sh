#!/bin/bash -e

if [ "$#" -ne 1 ]; then
    echo "usage: ${0} <path to docs>"
    exit 1
fi

path_to_docs=$1

cd ${path_to_docs}
make clean
make html

cd _build/html
tar cvzf ../html.tar.gz *
