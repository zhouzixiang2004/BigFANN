#include "utils.hpp"
#include "graph.hpp"
#include "gen_synthetic_filters.hpp"
#include <algorithm>
#include <filesystem>
#include <map>
using namespace std;

string DATA_DIR = "./data/";

int main() {
    size_t num_cols = 0;
    for (string file : {"arxiv/labels_diskann.txt","arxiv/query_labels_diskann.txt"}) {
        ifstream diskann(DATA_DIR + file);

        // Step 1: Load data
        std::vector<std::vector<int>> filters;
        std::string line;

        while (std::getline(diskann, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::vector<int> row;
            std::string value;

            while (std::getline(ss, value, ',')) {
                if (!value.empty()) {
                    row.push_back(std::stoi(value));
                }
            }

            filters.push_back(std::move(row));
        }

        // Step 2: Convert to SPMAT format
        std::vector<size_t> indptr;
        std::vector<vidType> indices;
        indptr.push_back(0);

        for (const auto& row : filters) {
            for (int f : row)
                indices.push_back(f);
            indptr.push_back(indices.size());
        }

        size_t num_rows = filters.size();
        for (const auto& row : filters)
            for (int f : row)
                if ((size_t)f + 1 > num_cols) num_cols = f + 1;
        size_t nnz = indices.size();
        cout << num_rows << " " << num_cols << " " << nnz << endl;

        // Step 3: Write spmat
        string output_file = "?";
        if (file == "arxiv/labels_diskann.txt") output_file = DATA_DIR + "arxiv/base.metadata.spmat";
        else output_file = DATA_DIR + "arxiv/query.metadata.spmat";
        write_spbitmat(output_file.c_str(),
                    num_rows,
                    num_cols,
                    nnz,
                    indptr.data(),
                    indices.data(),
                    true);
        
        if (file == "arxiv/labels_diskann.txt") output_file = DATA_DIR + "arxiv/labels_ung.txt";
        else output_file = DATA_DIR + "arxiv/query_labels_ung.txt";
        write_ung(filters,output_file,nnz);
    }

    return 0;
}
