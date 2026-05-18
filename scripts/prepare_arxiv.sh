#!/usr/bin/env bash

set -euo pipefail

if [ ! -d "data/arxiv" ]; then
    python ./scripts/download_arxiv.py
fi

DATA_DIR="data"
DATASET="arxiv"

export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32

g++ -pthread src/vec_to_bin.cpp -o bin/vec_to_bin -std=c++17
cd src
make convert_arxiv
cd ..

./bin/vec_to_bin float $DATA_DIR/$DATASET/query_vectors.fvecs $DATA_DIR/$DATASET/arxiv_query.fbin
./bin/vec_to_bin int $DATA_DIR/$DATASET/ground_truth_emis.ivecs $DATA_DIR/$DATASET/GT.ibin
./bin/vec_to_bin float $DATA_DIR/$DATASET/database_vectors.fvecs $DATA_DIR/$DATASET/arxiv_base.fbin
./bin/convert_arxiv