#include "utils.hpp"
#include "graph.hpp"
#include <algorithm>
#include <filesystem>
#include <map>
using namespace std;

string DATA_DIR = "./data/";

void usage(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s "
            "<dataset>\n"
            , argv[0]);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
    usage(argc, argv);
    string dataset_name = argv[1];
    cout << "running prepare_nhq " << dataset_name << endl;

    int N,A;
    // use same poss map for base and query
    vector<map<string,int> > poss;
    {
    ifstream data_labels(DATA_DIR + dataset_name + "_label/label_" + dataset_name + "_base.txt");
    data_labels >> N >> A;
    poss.resize(A);
    vector<vector<string> > labels(N,vector<string>(A));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < A; j++) {
            data_labels >> labels[i][j];
            if (!poss[j].count(labels[i][j])) {
                int c = poss[j].size();
                poss[j][labels[i][j]] = c;
            }
        }
    }
    int num_filters = 1;
    for (int i = 0; i < A; i++) num_filters *= poss[i].size();
    vector<size_t> indptr;
    for (int i = 0; i <= N; i++) indptr.push_back(i);
    vector<vidType> indices;
    for (int i = 0; i < N; i++) {
        int f = 0;
        for (int j = 0; j < A; j++) {
            f *= poss[j].size();
            f += poss[j][labels[i][j]];
        }
        indices.push_back(f);
    }
    write_spbitmat((DATA_DIR+dataset_name+"/base.metadata.spmat").c_str(),
        N,num_filters,N,indptr.data(),indices.data(),true);
    ofstream diskann(DATA_DIR+dataset_name+"/labels_diskann.txt");
    for (int i = 0; i < N; i++) diskann << indices[i] << endl;
    ofstream ung(DATA_DIR+dataset_name+"/labels_ung.txt");
    for (int i = 0; i < N; i++) ung << indices[i]+1 << endl;
    }
    
    {
    ifstream query_labels(DATA_DIR + dataset_name + "_label/label_" + dataset_name + "_query.txt");
    query_labels >> N >> A;
    vector<vector<string> > labels(N,vector<string>(A));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < A; j++) {
            query_labels >> labels[i][j];
            assert(poss[j].count(labels[i][j]));
        }
    }
    int num_filters = 1;
    for (int i = 0; i < A; i++) num_filters *= poss[i].size();
    vector<size_t> indptr;
    for (int i = 0; i <= N; i++) indptr.push_back(i);
    vector<vidType> indices;
    for (int i = 0; i < N; i++) {
        int f = 0;
        for (int j = 0; j < A; j++) {
            f *= poss[j].size();
            f += poss[j][labels[i][j]];
        }
        indices.push_back(f);
    }
    write_spbitmat((DATA_DIR+dataset_name+"/query.metadata.spmat").c_str(),
        N,num_filters,N,indptr.data(),indices.data(),true);
    ofstream diskann(DATA_DIR+dataset_name+"/query_labels_diskann.txt");
    for (int i = 0; i < N; i++) diskann << indices[i] << endl;
    ofstream ung(DATA_DIR+dataset_name+"/query_labels_ung.txt");
    for (int i = 0; i < N; i++) ung << indices[i]+1 << endl;
    }

    return 0;
}
