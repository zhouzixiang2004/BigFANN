#!/usr/bin/env bash

set -euo pipefail

bash scripts/prepare_nhq_datasets.sh

export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32
make -j

for DATASET in gist paper
do
    echo "./bin/build_index $DATASET hybrid 32 1 | tee logs/${DATASET}_build.txt"
    ./bin/build_index $DATASET hybrid 32 1 | tee logs/${DATASET}_build.txt
    echo "./bin/run_query $DATASET hybrid 10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500 750 1000 | tee logs/${DATASET}_query.txt"
    ./bin/run_query $DATASET hybrid 10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500 750 1000 | tee logs/${DATASET}_query.txt
done