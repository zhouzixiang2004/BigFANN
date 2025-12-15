#include "ivf.hpp"
#include "build_hybrid_graph.hpp"
#include "params.hpp"
#include "distance.hpp"
#include "parlay/primitives.h"
#include <filesystem>
using namespace std;

string DATA_DIR = "./data/";

template<typename T>
void build_hybrid_index(string data_file, string filters_file, 
    string index_folder, BuildParams P) {

    vector_dataset<T> data(data_file.c_str());
    sparse_bitmatrix filters(filters_file.c_str());
    if (P.INNER_PRODUCT) data.INNER_PRODUCT = true;

    vector<size_t> big;
    sparse_bitmatrix filterst = transpose(filters);
    for (size_t i = 0; i < filterst.rows; i++) {
        if (filterst[i].size() > (size_t)P.BIG_CUTOFF) big.push_back(i);
    }
    cout << "big: " << big.size() << endl;

    vector<Graph*> single;
    Graph* shared;
    vector<vid_t> init_vertex;
    build_hybrid_graph(data,filters,filterst,single,shared,init_vertex,P);

    filesystem::create_directory(index_folder);
    ofstream init(index_folder + "/init_vertex.ibin",ios::binary);
    int siz = init_vertex.size();
    init.write((char*)&siz, 4);
    init.write((char*)init_vertex.data(), sizeof(vid_t)*siz);
    init.close();

    string out_file = index_folder + "/shared";
    printf("Writing %s to disk file\n", out_file.c_str());
    shared->write_to_file(out_file);
    printf("Done writing to disk file\n");
    ofstream meta(out_file + ".meta.txt");
    meta << shared->V() << endl;
    meta << shared->E() << endl;
    meta << "4 8 1 4" << endl;
    meta << P.SHARED_DEGREE << endl;
    meta << "0" << endl;
    meta << "0" << endl;
    meta << "0" << endl;
    
    for (size_t i = 0; i < big.size(); i++) {
        int D = -1;
        for (auto [ub,d]: P.GRAPH_DEGREE) {
            if ((D == -1) && (filterst[big[i]].size() <= size_t(ub))) D = d;
        }
        assert(D != -1);

        string out_file = index_folder + "/single-" + to_string(i);
        printf("Writing %s to disk file\n", out_file.c_str());
        single[i]->write_to_file(out_file);
        printf("Done writing to disk file\n");
        ofstream meta(out_file + ".meta.txt");
        meta << single[i]->V() << endl;
        meta << single[i]->E() << endl;
        meta << "4 8 1 4" << endl;
        meta << D << endl;
        meta << "0" << endl;
        meta << "0" << endl;
        meta << "0" << endl;
    }
}

template<typename T>
void build_multifilter_index(string data_file, string filters_file, 
    string index_folder, BuildParams P) {
    
    vector_dataset<T> data(data_file.c_str());
    sparse_bitmatrix filters(filters_file.c_str());
    if (P.INNER_PRODUCT) data.INNER_PRODUCT = true;

    vector<size_t> big;
    sparse_bitmatrix filterst = transpose(filters);
    for (size_t i = 0; i < filterst.rows; i++) {
        if (filterst[i].size() > (size_t)P.BIG_CUTOFF) big.push_back(i);
    }
    cout << "big: " << big.size() << endl;

    vector<size_t> indptr;
    indptr.push_back(0);
    vector<vidType> indices;
    size_t nnz = 0;
    vector<uint64_t> bits((data.num+63)/64);
    for (size_t i = 0; i < big.size(); i++) {
        cout << i << " " << nnz << endl;
        for (auto x: filterst[big[i]]) bits[x >> 6] |= (1ULL << (x & 63));
        for (size_t j = i+1; j < big.size(); j++) {
            int num = 0;
            for (auto x: filterst[big[j]]) {
                if (bits[x >> 6] & (1ULL << (x & 63))) indices.push_back(x),num++;
            }
            nnz += num;
            indptr.push_back(nnz);
        }
        for (auto x: filterst[big[i]]) bits[x >> 6] = 0;
    }
    cout << "size: " << nnz << endl;
    filesystem::create_directory(index_folder);
    write_spbitmat((index_folder + "/precomp").c_str(),
        big.size()*(big.size()-1)/2,data.num,nnz,indptr.data(),indices.data());
    
    int done = 0,total = 0;
    for (size_t i = 0; i < big.size(); i++) total += filterst[big[i]].size();
    double total_time = 0;
    for (size_t i = 0; i < big.size(); i++) {
        Timer t;
        t.Start();
        cout << i << " " << filterst[big[i]].size() << endl;
        Graph g;
        vector<vid_t> group;
        for (auto x: filterst[big[i]]) group.push_back(x);
        
        vector_dataset<T> centroids;
        vector<int> membership;
        kmeans_cluster(data,group,(group.size()+P.CLUSTER_SIZE-1)/P.CLUSTER_SIZE,centroids,membership);
        assert(membership.size() == group.size());
        assert(centroids.num == (group.size()+P.CLUSTER_SIZE-1)/P.CLUSTER_SIZE);
        assert(centroids.dim == data.dim);
        std::ofstream mem(index_folder + "/ivf-" + to_string(i) + "-mem", std::ios::binary);
        mem.write((char*) membership.data(),4*membership.size());
        mem.close();
        write_vectors(centroids.num,centroids.dim,(index_folder + "/ivf-" + to_string(i) + "-cen").c_str(),centroids.data);
        
        t.Stop();
        total_time += t.Seconds();
        done += group.size();
        cout << "used time " << total_time << " for " << done << endl;
        cout << "total is " << total << endl;
        cout << "estimate need " << total_time*(total-done)/done << " more time" << endl;
    }
}

void usage(int argc, char *argv[]) {
  if (argc < 5) {
    fprintf(stderr, "Usage: %s "
            "<dataset> "
            "<folder> "
            "<single-deg> "
            "<shared-deg>\n"
            , argv[0]);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
    usage(argc, argv);
    string dataset_name = argv[1];
    string index_folder = argv[2];
    index_folder = DATA_DIR + dataset_name + "/" + index_folder;

    string vector_type;
    string data_file,filters_file;
    BuildParams P;
    if (dataset_name == "yfcc100M") {
        data_file = "base.10M.u8bin.crop_nb_10000000";
        filters_file = "base.metadata.10M.spmat";
        vector_type = "uint8";
        P.BIG_CUTOFF = 5000;
        P.CLUSTER_SIZE = 1000;
        P.BUILD_BEAM_WIDTH = 200;
        P.GRAPH_DEGREE = {{100000,8},{400000,10},{1e7,12}};
        P.SHARED_DEGREE = 1;
        P.ALPHA = 1.175;
    }
    else {
        data_file = dataset_name + "_base.fbin";
        filters_file = "base.metadata.spmat";
        vector_type = "float";
        if (dataset_name == "sift") {
            P.BIG_CUTOFF = 0;
            P.CLUSTER_SIZE = 256;
            P.BUILD_BEAM_WIDTH = 200;
            P.GRAPH_DEGREE = {{1e7,32}};
            P.SHARED_DEGREE = 1;
            P.ALPHA = 1.2;
        }
        else if (dataset_name == "sift100m") {
            data_file = dataset_name + "_base.u8bin";
            vector_type = "uint8";
            P.BIG_CUTOFF = 0;
            P.CLUSTER_SIZE = 100000;
            P.BUILD_BEAM_WIDTH = 200;
            P.GRAPH_DEGREE = {{1e8,6}};
            P.SHARED_DEGREE = 20;
            P.ALPHA = 1.2;
        }
        else if (dataset_name == "gist") {
            P.BIG_CUTOFF = 0;
            P.CLUSTER_SIZE = 256;
            P.BUILD_BEAM_WIDTH = 200;
            P.GRAPH_DEGREE = {{1e7,32}};
            P.ALPHA = 1.2;
        }
        else if (dataset_name == "paper") {
            P.BIG_CUTOFF = 0;
            P.CLUSTER_SIZE = 256;
            P.BUILD_BEAM_WIDTH = 200;
            P.GRAPH_DEGREE = {{1e7,32}};
            P.ALPHA = 1.2;
        }
        else if (dataset_name == "arxiv") {
            P.BIG_CUTOFF = 0;
            P.CLUSTER_SIZE = 256;
            P.BUILD_BEAM_WIDTH = 200;
            P.GRAPH_DEGREE = {{1e7,32}};
            P.SHARED_DEGREE = 1;
            P.ALPHA = 1.2;
        }
        else fprintf(stderr,"Unknown dataset"),exit(EXIT_FAILURE);
    }
    data_file = DATA_DIR + dataset_name + "/" + data_file;
    filters_file = DATA_DIR + dataset_name + "/" + filters_file;

    if (dataset_name != "yfcc100M") {
        P.GRAPH_DEGREE = {{1e9,stoi(argv[3])}};
        P.SHARED_DEGREE = stoi(argv[4]);
        cout << "Building with single degree " << stoi(argv[3]) << ", shared degree " << P.SHARED_DEGREE << endl;
    }
    
    if (vector_type == "uint8") {
        build_hybrid_index<uint8_t>(data_file,filters_file,index_folder,P);
        build_multifilter_index<uint8_t>(data_file,filters_file,index_folder,P);
    }
    else {
        build_hybrid_index<float>(data_file,filters_file,index_folder,P);
        build_multifilter_index<float>(data_file,filters_file,index_folder,P);
    }

    return 0;
}
