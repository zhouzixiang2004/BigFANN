#!/usr/bin/env bash

set -euo pipefail

# Configure parallelism and build
export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32
make -j

# Download the dataset
if [ ! -d "data/yfcc100M" ]; then
    python scripts/create_dataset.py --dataset yfcc-10M
fi

# Convert the dataset
./bin/convert_yfcc

# Build the index
./bin/build_index yfcc100M hybrid 10 1

RUNS=5   # number of repetitions
BEAMS=(10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500)

# Run the queries
for ((r=1; r<=RUNS; r++)); do\
    ./bin/run_query yfcc100M hybrid "${BEAMS[@]}"
done