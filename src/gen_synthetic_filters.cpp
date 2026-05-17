#include "utils.hpp"
#include "gen_synthetic_filters.hpp"
#include <algorithm>
#include <filesystem>
using namespace std;

string DATA_DIR = "./data/";

void usage(int argc, char *argv[]) {
  if (argc != 6) {
    fprintf(stderr,
      "Usage: %s <dataset> <dataset_num> <queries_num> "
      "<labels_per_node> <distinct_labels>\n",
      argv[0]);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
    usage(argc, argv);

    string dataset_name = argv[1];
    size_t dataset_num = stoull(argv[2]);
    size_t queries_num = stoull(argv[3]);
    int labels_per_node = stoi(argv[4]);
    int distinct_labels = stoi(argv[5]);

    double prob = (double) labels_per_node / distinct_labels;

    vector<vector<int> > labels = gen_uniform_prob(dataset_num, distinct_labels, prob);
    for (auto &l: labels) sort(l.begin(), l.end());

    vector<vector<int> > query_labels = gen_uniform(queries_num, distinct_labels, 1, 1);
    write_diskann(labels, DATA_DIR + dataset_name + "/labels_diskann.txt");
    write_diskann(query_labels, DATA_DIR + dataset_name + "/single_query_labels_diskann.txt");
    write_ung(labels, DATA_DIR + dataset_name + "/labels_ung.txt", distinct_labels);
    write_ung(query_labels, DATA_DIR + dataset_name + "/single_query_labels_ung.txt", distinct_labels);
    write_ours(labels, DATA_DIR + dataset_name + "/base.metadata.spmat", distinct_labels);
    write_ours(query_labels, DATA_DIR + dataset_name + "/single_query.metadata.spmat", distinct_labels);

    vector<vector<int> > query_labels_2 = gen_uniform(queries_num, distinct_labels, 2, 2);
    write_diskann(query_labels_2, DATA_DIR + dataset_name + "/double_query_labels_diskann.txt");
    write_ung(query_labels_2, DATA_DIR + dataset_name + "/double_query_labels_ung.txt", distinct_labels);
    write_ours(query_labels_2, DATA_DIR + dataset_name + "/double_query.metadata.spmat", distinct_labels);

    return 0;
}