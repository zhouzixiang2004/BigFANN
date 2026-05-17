#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32

for DATASET in gist paper
do
    echo "./bin/build_index $DATASET hybrid 32 1 | tee logs/${DATASET}_build.txt"
    ./bin/build_index $DATASET hybrid 32 1 | tee logs/${DATASET}_build.txt
    echo "./bin/run_query $DATASET hybrid 10 20 30 40 50 60 70 80 90 100 | tee logs/${DATASET}_query.txt"
    ./bin/run_query $DATASET hybrid 10 20 30 40 50 60 70 80 90 100 | tee logs/${DATASET}_query.txt
done