#!/usr/bin/env bash

set -euo pipefail

# Configure parallelism and build
export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32
make -j

# Download the dataset
if [ ! -d "data/arxiv" ]; then
    python ./scripts/download_arxiv.py
fi

# Prepare the dataset
bash scripts/prepare_arxiv.sh

# Build the index
./bin/build_index arxiv hybrid 32 1

RUNS=5   # number of repetitions
BEAMS=(10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500)

# Run the queries
for ((r=1; r<=RUNS; r++)); do\
    ./bin/run_query arxiv hybrid "${BEAMS[@]}"
done