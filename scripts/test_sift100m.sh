#!/usr/bin/env bash

set -euo pipefail

# Configure parallelism and build
export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32
make -j

# Download and prepare the dataset
bash scripts/prepare_sift100m.sh

# Convert the dataset
./bin/convert_sift100m

# Build the index
./bin/build_index sift100m hybrid_6_20 6 20

# Run single-filter queries
./bin/run_query sift100m hybrid_6_20 10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500
# Run double-filter queries
./bin/run_query sift100m-double hybrid_6_20 1 2 3 4 5 7 10 15 20 25 30 40 50 70 100