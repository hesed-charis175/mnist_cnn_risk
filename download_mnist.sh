#!/usr/bin/env bash
set -e
mkdir -p mnist_data && cd mnist_data
base="https://raw.githubusercontent.com/fgnt/mnist/master"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    echo "Fetching $f..."
    curl -sL -o "$f.gz" "$base/$f.gz"
    gunzip -f "$f.gz"
done
echo "Done. Files are in $(pwd)/"
ls -la
