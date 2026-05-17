#!/usr/bin/env bash

set -euo pipefail

bash scripts/download_nhq_datasets.sh

DATA_DIR="data"

export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32

g++ -pthread src/vec_to_bin.cpp -o bin/vec_to_bin -std=c++17
g++ -pthread src/bin_to_vec.cpp -o bin/bin_to_vec -std=c++17
cd src
make prepare_nhq
make get_groundtruth
cd ..

for DATASET in gist paper
do
    echo ./bin/vec_to_bin float $DATA_DIR/$DATASET/$DATASET\_base.fvecs $DATA_DIR/$DATASET/$DATASET\_base.fbin
    ./bin/vec_to_bin float $DATA_DIR/$DATASET/$DATASET\_base.fvecs $DATA_DIR/$DATASET/$DATASET\_base.fbin
    echo ./bin/vec_to_bin float $DATA_DIR/$DATASET/$DATASET\_query.fvecs $DATA_DIR/$DATASET/$DATASET\_query.fbin
    ./bin/vec_to_bin float $DATA_DIR/$DATASET/$DATASET\_query.fvecs $DATA_DIR/$DATASET/$DATASET\_query.fbin
    echo ./bin/prepare_nhq $DATASET
    ./bin/prepare_nhq $DATASET
    echo ./bin/get_groundtruth float $DATASET ""
    ./bin/get_groundtruth float $DATASET ""
    echo ./bin/bin_to_vec int $DATA_DIR/$DATASET/GT.ibin $DATA_DIR/$DATASET/GT.ivecs
    ./bin/bin_to_vec int $DATA_DIR/$DATASET/GT.ibin $DATA_DIR/$DATASET/GT.ivecs
done