#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

# Remove CUDA forward-compatibility libraries from LD_LIBRARY_PATH
# Disable CUDA forward compatibility
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "compat" | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_FORWARD_COMPAT_DISABLE=1

# Recompile if necessary (DO NOT CHANGE!)
mkdir -p build
cd build
cmake ..
make -j4

## Basic run to map the first 40 reads in the reads.fa in batches of 10 reads
#nsys profile --stats=true ./phyloTree --maxReads 10000 --batchSize 2000 --numThreads 8 --kmerSize 11 --bottomK 8000 > my.txt
./phyloTree --maxReads 10000 --batchSize 2000 --numThreads 8 --kmerSize 11 --bottomK 8000 > my.txt