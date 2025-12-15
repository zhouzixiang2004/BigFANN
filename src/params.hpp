#pragma once
#include <vector>
using namespace std;

struct BuildParams {
    bool INNER_PRODUCT = false;
    int BIG_CUTOFF;
    int CLUSTER_SIZE; // average cluster size
    int BUILD_BEAM_WIDTH;
    vector<pair<int,int> > GRAPH_DEGREE; // degrees for exclusive graphs
    double ALPHA;
    // hybrid index
    int SHARED_DEGREE = 0;
    int MIN_SHARED = 2;
};

struct QueryParams {
    int K = 10;
    bool INNER_PRODUCT = false;
    int BIG_CUTOFF; // must be same as build
    int BITSET_CUTOFF = 0;
    int PRECOMP_CUTOFF = 0;
    int PF_DIST = 4;
    int QUERY_BEAM_WIDTH;
    int NUM_CLUSTERS; // number of clusters to check in ivf
};