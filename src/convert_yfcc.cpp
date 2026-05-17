#include "utils.hpp"
#include "graph.hpp"
#include <algorithm>
#include <filesystem>
#include <map>
using namespace std;

string DATA_DIR = "./data/";

// ------------------------------------------------------------
//    Helper: process any subset (single-filter, double-filter)
// ------------------------------------------------------------
void process_subset(
    const string &prefix,                          // e.g. "single_filter_query" or "double_filter_query"
    const vector<int> &subset,
    sparse_bitmatrix &query_filters,
    vector_dataset<unsigned char> &queries,
    vector_dataset<int> &gt,
    int max_filter,
    int num_filters
) {
    if (subset.empty()) {
        cout << "Skipping '" << prefix << "' because subset is empty.\n";
        return;
    }

    cout << "Processing " << subset.size() << " queries for subset '" << prefix << "'...\n";

    // --- Prepare output dataset of vectors ---
    vector_dataset<unsigned char> out_queries;
    out_queries.num = subset.size();
    out_queries.dim = queries.dim;
    out_queries.data = new unsigned char[out_queries.num * out_queries.dim];

    vector<size_t> indptr;
    vector<vidType> indices;
    indptr.push_back(0);

    string base = DATA_DIR + "yfcc100M/" + prefix;

    ofstream diskann(base + "_labels_diskann.txt");
    ofstream ung(base + "_labels_ung.txt");

    // Fill query vectors + labels + spmat info
    for (size_t i = 0; i < subset.size(); i++) {
        memcpy(out_queries[i], queries[subset[i]], out_queries.dim);

        bool first = true;
        for (int f : query_filters[subset[i]]) {
            if (!first) { diskann << ","; ung << ","; }
            diskann << f;
            ung << (f + 1);
            first = false;

            indices.push_back(f);
        }
        diskann << "\n";
        ung << "\n";
        indptr.push_back(indices.size());
    }

    // Write query vectors
    write_vectors(out_queries.num, out_queries.dim,
        (base + ".public.100K.u8bin").c_str(),
        out_queries.data);

    // Write query filters spmat
    write_spbitmat((base + ".metadata.public.100K.spmat").c_str(),
        subset.size(), num_filters, indices.size(),
        indptr.data(), indices.data(), true);


    // -------------------------
    // Write ground truth
    // -------------------------
    vector_dataset<int> out_gt;
    out_gt.num = subset.size();
    out_gt.dim = gt.dim;
    out_gt.data = new int[out_gt.num * gt.dim];

    pair<int, float>* ung_gt = new pair<int, float>[out_gt.num * gt.dim];

    for (size_t i = 0; i < subset.size(); i++) {
        memcpy(out_gt[i], gt[subset[i]], gt.dim * sizeof(int));
        for (int j = 0; j < gt.dim; j++) {
            ung_gt[i * gt.dim + j] = { out_gt[i][j], 0.0 };
        }
    }

    write_vectors(out_gt.num, out_gt.dim,
        (base + "_GT.public.ibin").c_str(),
        out_gt.data);

    ofstream fout(base + "_gt_ung.fbin", ios::binary);
    fout.write(reinterpret_cast<const char*>(ung_gt),
        out_gt.num * gt.dim * sizeof(pair<int, float>));

    cout << "Finished subset '" << prefix << "'\n";
}

// ------------------------------------------------------------
//                         main()
// ------------------------------------------------------------
int main() {
    sparse_bitmatrix data_filters((DATA_DIR + "yfcc100M/base.metadata.10M.spmat").c_str());
    sparse_bitmatrix query_filters((DATA_DIR + "yfcc100M/query.metadata.public.100K.spmat").c_str());

    // Find max filter ID
    int max_filter = 0;
    for (size_t i = 0; i < data_filters.rows; i++) {
        for (int f : data_filters[i]) max_filter = max(max_filter, f);
    }
    cout << "Detected max filter " << max_filter << endl;

    // ----------------------
    // Write dataset labels
    // ----------------------
    ofstream diskann(DATA_DIR + "yfcc100M/labels_diskann.txt");
    ofstream ung(DATA_DIR + "yfcc100M/labels_ung.txt");

    for (size_t i = 0; i < data_filters.rows; i++) {
        bool first = true;
        for (int f : data_filters[i]) {
            if (!first) { diskann << ","; ung << ","; }
            diskann << f;
            ung << f + 1;
            first = false;
        }
        if (first) {
            // no labels: patch required for diskann
            diskann << -1;
            ung << max_filter + 2;
        }
        diskann << "\n";
        ung << "\n";
    }

    // ---------------------------
    //       Split queries
    // ---------------------------
    vector<int> single_filter, double_filter, mixed_filter;

    for (size_t i = 0; i < query_filters.rows; i++) {
        if (query_filters[i].size() == 1) single_filter.push_back(i);
        if (query_filters[i].size() == 2) double_filter.push_back(i);
        mixed_filter.push_back(i);
    }

    sparse_bitmatrix filterst = transpose(data_filters);
    vector<int> specificity(query_filters.rows);
    #pragma omp parallel for
    for (size_t i = 0; i < query_filters.rows; i++) {
        if (query_filters[i].size() == 1) {
            specificity[i] = filterst[query_filters[i][0]].size();
        }
        else if (query_filters[i].size() == 2) {
            int a = query_filters[i][0];
            int b = query_filters[i][1];
            if (filterst[a].size() > filterst[b].size()) swap(a,b);
            int num_inter = 0;
            for (auto v: filterst[a]) {
                if (data_filters[v].contains(b)) num_inter++;
            }
            specificity[i] = num_inter;
        }
    }
    vector<int> order(query_filters.rows);
    for (size_t i = 0; i < order.size(); i++) order[i] = i;
    sort(order.begin(),order.end(),[&](int a,int b) { return specificity[a] < specificity[b]; });

    cout << "1pc specificity: " << specificity[order[(int)((double) 1/100*order.size())]] << "\n";
    cout << "25pc specificity: " << specificity[order[(int)((double) 25/100*order.size())]] << "\n";
    cout << "50pc specificity: " << specificity[order[(int)((double) 50/100*order.size())]] << "\n";
    cout << "75pc specificity: " << specificity[order[(int)((double) 75/100*order.size())]] << "\n";

    vector<int> pc1,pc25,pc50,pc75;
    for (size_t i = 0; i < order.size(); i++) {
        double percentile = (double) i / order.size() * 100;
        if (abs(percentile-1) <= 1) pc1.push_back(order[i]);
        if (abs(percentile-25) <= 1) pc25.push_back(order[i]);
        if (abs(percentile-50) <= 1) pc50.push_back(order[i]);
        if (abs(percentile-75) <= 1) pc75.push_back(order[i]);
    }

    cout << single_filter.size() << " single-filter queries\n";
    cout << double_filter.size() << " double-filter queries\n";
    cout << mixed_filter.size() << " mixed-filter queries\n";

    cout << pc1.size() << " 1pc queries\n";
    cout << pc25.size() << " 25pc queries\n";
    cout << pc50.size() << " 50pc queries\n";
    cout << pc75.size() << " 75pc queries\n";

    // Load query vectors + GT
    vector_dataset<unsigned char> queries(
        (DATA_DIR + "yfcc100M/query.public.100K.u8bin").c_str());
    vector_dataset<int> gt(
        (DATA_DIR + "yfcc100M/GT.public.ibin").c_str());
    
    vector_dataset<int> gt_k100(
        (DATA_DIR + "yfcc100M/k100_GT.ibin").c_str());
    pair<int, float>* ung_gt = new pair<int, float>[gt_k100.num * gt_k100.dim];
    for (size_t i = 0; i < gt_k100.num; i++) {
        for (int j = 0; j < gt_k100.dim; j++) {
            ung_gt[i * gt_k100.dim + j] = { gt_k100[i][j], 0.0 };
        }
    }
    ofstream fout(DATA_DIR + "yfcc100M/k100_gt_ung.fbin", ios::binary);
    fout.write(reinterpret_cast<const char*>(ung_gt),
        gt_k100.num * gt_k100.dim * sizeof(pair<int, float>));

    // ------------------------------------
    // Process the subsets
    // ------------------------------------
    process_subset("single_query", single_filter,
                   query_filters, queries, gt,
                   max_filter, query_filters.cols);

    process_subset("double_query", double_filter,
                   query_filters, queries, gt,
                   max_filter, query_filters.cols);
    
    process_subset("mixed_query", mixed_filter,
                   query_filters, queries, gt,
                   max_filter, query_filters.cols);
    
    process_subset("pc1_query", pc1,
                   query_filters, queries, gt,
                   max_filter, query_filters.cols);
    process_subset("pc25_query", pc25,
                   query_filters, queries, gt,
                   max_filter, query_filters.cols);
    process_subset("pc50_query", pc50,
                   query_filters, queries, gt,
                   max_filter, query_filters.cols);
    process_subset("pc75_query", pc75,
                   query_filters, queries, gt,
                   max_filter, query_filters.cols);

    cout << "All done.\n";
    return 0;
}
