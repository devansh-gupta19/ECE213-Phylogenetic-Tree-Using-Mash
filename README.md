# ECE213 Phylogenetic Tree Using MASH

GPU-accelerated phylogenetic tree construction using MASH distance metrics.

## Contents
* [Overview](#overview)
* [Project Structure](#project-structure)
* [Building and Running](#building-and-running)
* [Setup](#setup)

## Overview
This project implements a GPU-accelerated phylogenetic tree construction pipeline using MASH distance metrics and the neighbor joining algorithm. The implementation accelerates the computation of pairwise distances between DNA sequences using CUDA, enabling efficient phylogenetic analysis on the DIPPER dataset. The generated phylogenetic tree can be visualized in Newick format and analyzed to understand evolutionary relationships between species.

**Repository Branches**: This repository contains two branches:
- **main**: Contains the final paralellelized GPU-accelerated implementation
- **baseline_sequential**: Contains the sequential implementation used as the baseline for performance comparison

## Project Structure
- **src/**: Contains CUDA kernels and C++ source code for phylogenetic tree construction
  - `main.cpp`: Main entry point
  - `mash.cu`: MASH distance computation on GPU
  - `kmer.cu`: K-mer extraction and processing on GPU
  - `neighborJoining.cu`: Neighbor joining algorithm implementation
  - `murmur3Hash.cu`: Hash function for k-mer processing
  - `memTransfer.cu`: GPU memory transfer and data movement utilities
  - `twoBitCompressor.*`: DNA sequence compression utilities
  - `newickHelper.*`: Newick format tree output utilities
- **dataset_dipper/**: Contains the dipper dataset (DNA sequences)
- **CMakeLists.txt**: Build configuration
- **compare_nwk.py**: Python script for comparing phylogenetic trees in Newick format
- **data.treefile.nwk**: Reference phylogenetic tree in Newick format for validation (obtained from DIPPER dataset)
- **docs/**: Project documentation including proposal, report, slides, and presentation video


## Building and Running
The project has been developed using UC San Diego's Data Science/Machine Learning Platform ([DSMLP](https://blink.ucsd.edu/faculty/instruction/tech-guide/dsmlp/index.html)), which provides students with access to research-class CPU and GPU resources for coursework and projects.

To get set up, please follow the steps below:

1. SSH into the DSMLP server (dsmlp-login.ucsd.edu) using the AD account. I recommend using PUTTY SSH client (putty.org) or Windows Subsystem for Linux (WSL) for Windows (https://docs.microsoft.com/en-us/windows/wsl/install-manual). MacOS and Linux users can SSH into the server using the following command (replace `d6gupta` with your username)

```
ssh d6gupta@dsmlp-login.ucsd.edu
```

2. Next, clone the assignment repository in your HOME directory using the following example command:
```
cd ~
git clone https://github.com/devansh-gupta19/ECE260B-Dual-Core-ML-Accelerator.git
cd ~
```

3. The source code directory (in the `src/` directory) has a `run-commands.sh` script which contains commands that will be executed via the Docker container on the GPU instance. You can modify the commands of this script depending on the experiment.
```
cd ECE213-Phylogenetic-Tree-Using-Mash
ssh d6gupta@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -v a30 -c 8 -g 1 -m 8 -i yatisht/ece213-wi26:latest -f ./runCommands.sh
```
