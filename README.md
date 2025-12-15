# BigFANN

First, to build the project, run `make -j` from the `BigFANN` directory.

To generate synthetic labels for and evaluate the `sift1m` dataset, run the following commands:

```
export OMP_NUM_THREADS=32
export PARLAY_NUM_THREADS=32
make -j

# Download dataset
cd ./data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -zxvf sift.tar.gz
cd ..

# Convert vector data
./bin/vec_to_bin float ./data/sift/sift_base.fvecs ./data/sift/sift_base.fbin
./bin/vec_to_bin float ./data/sift/sift_query.fvecs ./data/sift/sift_query.fbin

# Generate synthetic labels
./bin/gen_synthetic_filters 

# Get ground truth
./bin/get_groundtruth float sift single_
./bin/get_groundtruth float sift double_

# Build index
./bin/build_index sift hybrid_32_1 32 1

# Run single-filter queries
./bin/run_query sift hybrid_32_1 10 20 30 40 50 60 70 80 90 100
# Run double-filter queries
./bin/run_query sift-double hybrid_32_1 1 2 3 4 5 7 10 15 20 25 30 40 50 70 100
```