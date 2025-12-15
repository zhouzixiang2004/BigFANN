#include "utils.hpp"
#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
using namespace std;

vector<int> gen_distinct(int l,int r,int num,mt19937 &gen) {
    assert(l <= r);
    assert(num <= r-l+1);
    if (num < (r-l+1)/2) {
        vector<int> ret;
        vector<uint64_t> seen((r-l+1)/64+1);
        auto dist = uniform_int_distribution<int>(l,r);
        while (ret.size() < size_t(num)) {
            int u = dist(gen);
            if (!(seen[(u-l) >> 6] & (1ULL << ((u-l) & 63)))) {
                ret.push_back(u);
                seen[(u-l) >> 6] |= (1ULL << ((u-l) & 63));
            }
        }
        return ret;
    }
    else {
        vector<int> ret;
        for (int i = l; i <= r; i++) ret.push_back(i);
        shuffle(ret.begin(),ret.end(),gen);
        ret.resize(num);
        return ret;
    }
}

vector<int> gen_distinct_weighted(int num,const vector<double> &weights,mt19937 &gen) {
    vector<int> ret;
    vector<uint64_t> seen(weights.size()/64+1);
    auto dist = discrete_distribution<int>(weights.begin(),weights.end());
    while (ret.size() < size_t(num)) {
        int u = dist(gen);
        if (!(seen[u >> 6] & (1ULL << (u & 63)))) {
            ret.push_back(u);
            seen[u >> 6] |= (1ULL << (u & 63));
        }
    }
    return ret;
}

vector<vector<int> > gen_uniform(int N,int distinct_labels,
    int min_labels_per_node = 0,int max_labels_per_node = -1) {
    // For each node, first randomly choose how many labels it has uniformly in [min,max]
    // Then pick this many labels uniformly at random
    
    if (max_labels_per_node == -1) max_labels_per_node = distinct_labels;

    mt19937 gen;
    vector<vector<int> > filters(N);
    for (int i = 0; i < N; i++) {
        int num_labels = uniform_int_distribution<int>(min_labels_per_node,max_labels_per_node)(gen);
        filters[i] = gen_distinct(0,distinct_labels-1,num_labels,gen);
    }
    return filters;
}

vector<vector<int>> gen_uniform_prob(int N,int distinct_labels,double prob) {
    // Each (node, label) has a probability 'prob' of being present
    mt19937 gen;
    bernoulli_distribution dist(prob);

    vector<vector<int> > filters(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < distinct_labels; j++) {
            if (dist(gen)) filters[i].push_back(j);
        }
    }
    return filters;
}

vector<vector<int> > gen_zipf(int N,int distinct_labels,
    int min_labels_per_node = 0,int max_labels_per_node = -1) {
    // For each node, first randomly choose how many labels it has uniformly in [min,max]
    // Then pick this many labels with weights proportional to Zipf distribution

    if (max_labels_per_node == -1) max_labels_per_node = distinct_labels;

    mt19937 gen;
    vector<vector<int> > filters(N);
    vector<double> weights(distinct_labels);
    for (int i = 0; i < distinct_labels; i++) weights[i] = 1.0/(i+1);
    for (int i = 0; i < N; i++) {
        int num_labels = uniform_int_distribution<int>(min_labels_per_node,max_labels_per_node)(gen);
        filters[i] = gen_distinct_weighted(num_labels,weights,gen);
    }
    return filters;
}

// Writes labels for DiskANN format
void write_diskann(const vector<vector<int>>& filters, const string& filepath) {
    ofstream diskann(filepath);
    for (size_t i = 0; i < filters.size(); i++) {
        bool first = true;
        for (int f : filters[i]) {
            if (!first) diskann << ",";
            diskann << f;
            first = false;
        }
        if (first) { // empty filter
            diskann << -1;
        }
        diskann << "\n";
    }
}

// Writes labels for UNG format
void write_ung(const vector<vector<int>>& filters, const string& filepath, size_t num_filters) {
    ofstream ung(filepath);
    for (size_t i = 0; i < filters.size(); i++) {
        bool first = true;
        for (int f : filters[i]) {
            if (!first) ung << ",";
            ung << f + 1;  // note the +1 shift
            first = false;
        }
        if (first) { // empty filter
            ung << num_filters + 1;
        }
        ung << "\n";
    }
}

// Writes labels for SPMAT format
void write_ours(const vector<vector<int>>& filters, const string& filepath, size_t num_filters) {
    vector<size_t> indptr;
    vector<vidType> indices;
    indptr.reserve(filters.size() + 1);
    indptr.push_back(0);

    for (const auto& row : filters) {
        indptr.push_back(indptr.back() + row.size());
        indices.insert(indices.end(), row.begin(), row.end());
    }

    write_spbitmat(filepath.c_str(),
                   filters.size(),
                   num_filters,
                   indptr.back(),
                   indptr.data(),
                   indices.data(),
                   true);
}
