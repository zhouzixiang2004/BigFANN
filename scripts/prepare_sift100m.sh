#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="data"

export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32

g++ -pthread src/vec_to_bin.cpp -o bin/vec_to_bin -std=c++17
cd src
make gen_synthetic_filters
make get_groundtruth
cd ..

# Download the dataset
if [ ! -f "$DATA_DIR/bigann_query.bvecs" ]; then
    echo "Downloading BIGANN query set..."
    wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz -O "$DATA_DIR/bigann_query.bvecs.gz"
    gunzip -f "$DATA_DIR/bigann_query.bvecs.gz"
fi
if [ ! -f "$DATA_DIR/bigann_base.bvecs" ]; then
    echo "Downloading BIGANN base set..."
    wget -c ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz -O "$DATA_DIR/bigann_base.bvecs.gz"
    gunzip -f "$DATA_DIR/bigann_base.bvecs.gz"
fi

DATASET="sift100m"
DSIZE=100000000
mkdir $DATA_DIR/$DATASET
./bin/vec_to_bin uint8 $DATA_DIR/bigann_base.bvecs $DATA_DIR/$DATASET/$DATASET\_base.u8bin $DSIZE
./bin/vec_to_bin uint8 $DATA_DIR/bigann_query.bvecs $DATA_DIR/$DATASET/$DATASET\_query.u8bin
./bin/gen_synthetic_filters $DATASET $DSIZE 10000 8 20
./bin/get_groundtruth uint8 $DATASET single_
./bin/get_groundtruth uint8 $DATASET double_