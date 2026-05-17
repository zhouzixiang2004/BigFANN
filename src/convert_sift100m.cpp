#include "utils.hpp"
#include "graph.hpp"
#include <algorithm>
#include <filesystem>
#include <map>
using namespace std;

string DATA_DIR = "./data/";

int main() {
    sparse_bitmatrix data_filters((DATA_DIR + "sift100m/base.metadata.spmat").c_str());

    int max_filter = 0;
    for (size_t i = 0; i < data_filters.rows; i++) {
        for (int f: data_filters[i]) max_filter = max(max_filter,f);
    }
    cout << "Detected max filter " << max_filter << endl;

    ofstream diskann(DATA_DIR + "sift100m/labels_diskann.txt");
    ofstream ung(DATA_DIR + "sift100m/labels_ung.txt");
    for (size_t i = 0; i < data_filters.rows; i++) {
        bool first = true;
        for (int f: data_filters[i]) {
            if (!first) diskann << ",",ung << ",";
            diskann << f,ung << f+1,first = false;
        }
        // somehow there exist vectors with no labels, which breaks diskann
        if (first) diskann << -1,ung << max_filter+2;
        diskann << endl,ung << endl;
    }

    vector_dataset<unsigned char> queries((DATA_DIR + "sift100m/sift100m_query.u8bin").c_str());
    for (string prefix: {"single_","double_"}) {
        sparse_bitmatrix query_filters((DATA_DIR + "sift100m/" + prefix + "query.metadata.spmat").c_str());
        vector_dataset<int> gt((DATA_DIR + "sift100m/" + prefix + "GT.ibin").c_str());
        
        ofstream diskann2(DATA_DIR + "sift100m/" + prefix + "query_labels_diskann.txt");
        ofstream ung2(DATA_DIR + "sift100m/" + prefix + "query_labels_ung.txt");
        for (size_t i = 0; i < query_filters.rows; i++) {
            bool first = true;
            for (int f: query_filters[i]) {
                if (!first) diskann2 << ",",ung2 << ",";
                diskann2 << f,ung2 << f+1,first = false;
            }
            assert(query_filters[i].size() <= 2);
            if (first) diskann2 << -1,ung2 << max_filter+2;
            diskann2 << endl,ung2 << endl;
        }
        
        pair<int,float> *ung_gt = new pair<int,float>[query_filters.rows*gt.dim];
        for (size_t i = 0; i < query_filters.rows; i++) {
            for (int j = 0; j < gt.dim; j++) ung_gt[i*gt.dim+j] = make_pair(gt[i][j],0.0);
        }
        ofstream fout(DATA_DIR + "sift100m/" + prefix + "_gt_ung.fbin", std::ios::binary);
        fout.write(reinterpret_cast<const char*>(ung_gt), query_filters.rows*gt.dim*sizeof(std::pair<int, float>));
    }

    return 0;
}
