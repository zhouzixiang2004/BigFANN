#include "utils.hpp"
#include "gen_synthetic_filters.hpp"
#include <algorithm>
#include <filesystem>
using namespace std;

string DATA_DIR = "./data/";

int main() {
    string dataset_name = "sift";
    vector_dataset<float> data((DATA_DIR + dataset_name + "/"
        + dataset_name + "_base.fbin").c_str());
    vector_dataset<float> queries((DATA_DIR + dataset_name + "/"
        + dataset_name + "_query.fbin").c_str());
    
    for (auto [labels_per_node,distinct_labels]: {pair{8,20}}) {
        double prob = (double) labels_per_node / distinct_labels;

        // Generate random labels
        vector<vector<int> > labels = 
            gen_uniform_prob(data.num,distinct_labels,prob);
        for (auto &l: labels) sort(l.begin(),l.end());
        vector<vector<int> > query_labels =
            gen_uniform(queries.num,distinct_labels,1,1);
        
        write_diskann(labels,DATA_DIR + dataset_name + "/labels_diskann.txt");
        write_diskann(query_labels,DATA_DIR + dataset_name + "/single_query_labels_diskann.txt");
        write_ung(labels,DATA_DIR + dataset_name + "/labels_ung.txt",distinct_labels);
        write_ung(query_labels,DATA_DIR + dataset_name + "/single_query_labels_ung.txt",distinct_labels);
        write_ours(labels,DATA_DIR + dataset_name + "/base.metadata.spmat",distinct_labels);
        write_ours(query_labels,DATA_DIR + dataset_name + "/single_query.metadata.spmat",distinct_labels);

        vector<vector<int> > query_labels_2 =
            gen_uniform(queries.num,distinct_labels,2,2);
        
        write_diskann(query_labels_2,DATA_DIR + dataset_name + "/double_query_labels_diskann.txt");
        write_ung(query_labels_2,DATA_DIR + dataset_name + "/double_query_labels_ung.txt",distinct_labels);
        write_ours(query_labels_2,DATA_DIR + dataset_name + "/double_query.metadata.spmat",distinct_labels);
    }

    return 0;
}