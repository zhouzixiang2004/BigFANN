#include "utils.hpp"
#include "distance.hpp"
#include <omp.h>
#include <map>
using namespace std;

string DATA_DIR = "./data/";

void usage(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s "
            "<type (float or uint8)> <dataset> <prefix>\n"
            , argv[0]);
    exit(EXIT_FAILURE);
  }
}

const int KK = 10;
const int PF_DIST = 4;
template <typename T>
int getKNN(vector_dataset<T> &data,vector_dataset<int> &results,
    int query_id,T *query_data,vidType *candidates,int num) {
    
    int l = data.dim*sizeof(T)/64;
    vector<pair<float,vidType> > dists(KK,make_pair(1e99,-1));
    for (int i = 0; i < num; i++) {
        auto v = candidates[i];
        float d = compute_distance_squared(data.dim,data[v],query_data);
        if (d < dists.back().first) {
            dists.pop_back();
            dists.insert(lower_bound(dists.begin(),dists.end(),make_pair(d,v)),make_pair(d,v));
        }
        if (i+PF_DIST < num) {
            char *a = (char*) data[candidates[i+PF_DIST]];
            for (int j = 0; j < l; j++) __builtin_prefetch(a+64*j);
        }
    }
    if (dists.back().second == vidType(-1)) cerr << "Fewer than K found" << endl;
    for (int j = 0; j < KK; j++) results[query_id][j] = dists[j].second;
    return num;
}
template<typename T>
void get_groundtruth(string dataset_name,string suff="fbin",string pref="") {
    // Remember to check that these are the correct file names
    string data_file = dataset_name + "_base." + suff;
    string filters_file = "base.metadata.spmat";
    string queries_file = dataset_name + "_query." + suff;
    string query_filters_file = pref + "query.metadata.spmat";
    string gt_file = pref + "GT.ibin";

    data_file = DATA_DIR + dataset_name + "/" + data_file;
    filters_file = DATA_DIR + dataset_name + "/" + filters_file;
    queries_file = DATA_DIR + dataset_name + "/" + queries_file;
    query_filters_file = DATA_DIR + dataset_name + "/" + query_filters_file;
    gt_file = DATA_DIR + dataset_name + "/" + gt_file;

    vector_dataset<T> data(data_file.c_str());
    sparse_bitmatrix filters(filters_file.c_str());
    sparse_bitmatrix filterst = transpose(filters);
    vector_dataset<T> queries(queries_file.c_str());
    sparse_bitmatrix query_filters(query_filters_file.c_str());
    int N = data.num;

    int num_threads = 0;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    std::cout << "OpenMP get_groundtruth (" << num_threads << " threads)\n";

    vector<vector<uint64_t> > bitsets(filterst.rows);
    for (size_t i = 0; i < filterst.rows; i++) {
        if (filterst[i].size() > 10000) {
            bitsets[i].resize((N+63)/64);
            for (auto v: filterst[i]) bitsets[i][v >> 6] |= (1ULL << (v & 63));
        }
    }

    vector_dataset<int> results;
    results.num = queries.num;
    results.dim = KK;
    results.data = new int[queries.num*KK];
    #pragma omp parallel for schedule(dynamic,100)
    for (size_t i = 0; i < queries.num; i++) {
        int x = i;
        if ((i % 100) == 0) cout << i << endl;

        auto query_data = queries[x];
        if (query_filters[x].size() == 1) {
            // single filter
            int a = query_filters[x][0];
            getKNN(data,results,x,query_data,filterst.get_begin(a),
                filterst.get_size(a));
        }
        else {
            // multiple filters
            size_t s = 0;
            for (size_t j = 0; j < query_filters[x].size(); j++) {
                if (filterst[query_filters[x][j]].size() < filterst[query_filters[x][s]].size())
                    s = j;
            }
            // loop over everything in the smallest filter
            vector<vidType> c;
            for (auto v: filterst[query_filters[x][s]]) {
                bool bad = false;
                for (size_t j = 0; j < query_filters[x].size(); j++) {
                    if (!bitsets[query_filters[x][j]].empty()) {
                        if ((j != s) && !(bitsets[query_filters[x][j]][v >> 6] & (1ULL << (v & 63)))) {
                            bad = true;
                            break;
                        }
                    }
                    else {
                        if ((j != s) && !filterst[query_filters[x][j]].contains(v)) {
                            bad = true;
                            break;
                        }
                    }
                }
                if (!bad) c.push_back(v);
            }
            getKNN(data,results,x,query_data,c.data(),c.size());
        }
    }

    write_vectors(results.num,results.dim,gt_file.c_str(),results.data);
}

int main(int argc, char *argv[]) {
    usage(argc, argv);
    string type = argv[1];
    string dataset_name = argv[2];
    string pref = argv[3];
    if (type == "float") get_groundtruth<float>(dataset_name,"fbin",pref);
    else if (type == "uint8") get_groundtruth<unsigned char>(dataset_name,"u8bin",pref);
    else cout << "invalid type" << endl;
    return 0;
}
