#pragma once

#include <cmath>
#include <cassert>
#include <chrono>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <random>

using namespace std;

template <typename T>
inline void PREFETCH_VECTOR(int dim, T *vec) {
#ifdef USE_PF_VEC
    char* a = (char*)vec;
    int l = dim*sizeof(T)/64;
    for (int j = 0; j < l; j++) {
        __builtin_prefetch(a+64*j);
        //_mm_prefetch(a+64*j, _MM_HINT_T2);
        //if (j < 4) _mm_prefetch(a+64*j, _MM_HINT_T0);
        //else _mm_prefetch(a+64*j, _MM_HINT_T2);
    }
#endif
}

// fbin, ibin, u8bin
template <typename T>
void load_vectors(size_t &num, int &dim, const char *filename, T *&data) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    size_t nvecs = 0;
    in.read((char*)&nvecs, 4);
    if (num == 0) num = nvecs;
    in.read((char*)&dim, 4);
    if ((dim*sizeof(T)) % 32 == 0) {
        //cout << filename << " loading aligned" << endl;
        data = (T*)aligned_alloc(32,(int64_t) num*dim*sizeof(T));
    }
    else {
        //cout << filename << " loading unaligned" << endl;
        data = (T*)malloc((int64_t) num*dim*sizeof(T));
    }
    for (size_t i = 0; i < num; i++) in.read((char*)(data+i*dim), sizeof(T)*dim);
    in.close();
}

template <typename T>
void write_vectors(size_t num, int dim, const char *filename, T *data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    out.write((char*)&num, 4);
    out.write((char*)&dim, 4);
    for (size_t i = 0; i < num; i++) out.write((char*)(data+i*dim), sizeof(T)*dim);
    out.close();
}

template <typename T>
class vector_dataset {
    public:
        size_t num;
        int dim;
        T *data;
        bool INNER_PRODUCT = false;

        vector_dataset() { num = dim = 0; data = new T[1]; }
        ~vector_dataset() {
            delete[] data;
        }
        vector_dataset(const char *filename, size_t num = 0) {
            this->num = num;
            load_vectors(this->num,this->dim,filename,data);
        }
        T *operator[](size_t i) { return &data[i*dim]; }

        vector_dataset(const vector_dataset&) = delete;
        vector_dataset& operator=(const vector_dataset&) = delete;
        vector_dataset(vector_dataset&& other) noexcept {
            num = other.num;
            dim = other.dim;
            data = other.data;
            INNER_PRODUCT = other.INNER_PRODUCT;
            other.data = nullptr;
        }

        vector_dataset& operator=(vector_dataset&& other) noexcept {
            if (this != &other) {
                free(data);
                num = other.num;
                dim = other.dim;
                data = other.data;
                INNER_PRODUCT = other.INNER_PRODUCT;
                other.data = nullptr;
            }
            return *this;
        }
};

#include "VertexSet.hpp"
void load_spmat(const char *filename, size_t &rows, size_t &cols, \
    size_t &nnz, size_t *&indptr, vidType *&indices, float *&data) {
    
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    in.read((char*)&rows, 8);
    in.read((char*)&cols, 8);
    in.read((char*)&nnz, 8);
    indptr = new size_t[rows+1];
    in.read((char*)indptr,8*(rows+1));
    assert(indptr[rows] == nnz);
    indices = new vidType[nnz];
    in.read((char*)indices,4*nnz);
    data = new float[nnz];
    in.read((char*)data,4*nnz);
    in.close();
}

void load_spbitmat(const char *filename, size_t &rows, size_t &cols, \
    size_t &nnz, size_t *&indptr, vidType *&indices) {
    
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    in.read((char*)&rows, 8);
    in.read((char*)&cols, 8);
    in.read((char*)&nnz, 8);
    indptr = new size_t[rows+1];
    in.read((char*)indptr,8*(rows+1));
    assert(indptr[rows] == nnz);
    indices = new vidType[nnz];
    in.read((char*)indices,4*nnz);
    in.close();
}

void write_spbitmat(const char *filename, size_t rows, size_t cols, \
    size_t nnz, size_t *indptr, vidType *indices, bool extra_stuff = false) {
    
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    out.write((char*)&rows, 8);
    out.write((char*)&cols, 8);
    out.write((char*)&nnz, 8);
    out.write((char*)indptr,8*(rows+1));
    out.write((char*)indices,4*nnz);
    if (extra_stuff) {
        vector<float> data(nnz,1);
        out.write((char*) data.data(),4*nnz);
    }
    out.close();
}

class sparse_bitmatrix {
    public:
        size_t rows,cols,nnz;
        size_t *indptr;
        vidType *indices;

        sparse_bitmatrix() { rows = cols = nnz = 0; }
        ~sparse_bitmatrix() {
            delete[] indptr;
            delete[] indices;
        }
        sparse_bitmatrix(const char *filename,bool has_data = true) {
            if (has_data) {
                float *data;
                load_spmat(filename,rows,cols,nnz,indptr,indices,data);
                for (size_t i = 0; i < rows; i++) sort(indices+indptr[i],indices+indptr[i+1]);
                delete[] data;
            }
            else {
                load_spbitmat(filename,rows,cols,nnz,indptr,indices);
                // for (size_t i = 0; i < rows; i++) sort(indices+indptr[i],indices+indptr[i+1]);
            }
        }
        VertexSet operator[](size_t i) const {
            //assert(0 <= i && i < rows);
            return VertexSet(&indices[indptr[i]],vidType(indptr[i+1]-indptr[i]),vidType(i));
        }
        vidType *get_begin(size_t i) {
            return indices+indptr[i];
        }
        vidType *get_end(size_t i) {
            return indices+indptr[i+1];
        }
        int get_size(size_t i) {
            return indptr[i+1]-indptr[i];
        }
};

sparse_bitmatrix transpose(const sparse_bitmatrix &mat) {
    sparse_bitmatrix new_mat;
    new_mat.rows = mat.cols;
    new_mat.cols = mat.rows;
    new_mat.nnz = mat.nnz;
    new_mat.indptr = new size_t[new_mat.rows+1];
    memset(new_mat.indptr,0,8*(new_mat.rows+1));
    for (size_t i = 0; i < new_mat.nnz; i++) new_mat.indptr[mat.indices[i]]++;
    for (size_t i = 0; i < new_mat.rows; i++) new_mat.indptr[i+1] += new_mat.indptr[i];
    new_mat.indices = new vidType[new_mat.nnz];
    for (size_t i = mat.rows-1; ; i--) {
        for (size_t j = mat.indptr[i]; j < mat.indptr[i+1]; j++)
            new_mat.indices[--new_mat.indptr[mat.indices[j]]] = i;
        if (i == 0) break;
    }
    return new_mat;
}

double compute_avg_recall_1D(vector_dataset<int> &results, vector_dataset<int> &gt) {
    assert(results.num == gt.num);
    assert(results.dim <= gt.dim);
    size_t qsize = results.num;
    int K = results.dim;

    int64_t correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (size_t q_i = 0; q_i < qsize; ++q_i) {
        for (int top_i = 0; top_i < K; ++top_i) {
            auto true_id = gt[q_i][top_i];
            for (int n_i = 0; n_i < K; ++n_i) {
                if (results[q_i][n_i] == true_id) {
                    correct ++;
                    break;
                }
            }
        }
    }
    int64_t total = K * qsize;
    return double(correct) / double(total);
}

vector<int> join(const vector<int> &a,const vector<int> &b) {
    size_t i = 0,j = 0;
    vector<int> c;
    c.reserve(min(a.size(),b.size()));
    while ((i < a.size()) && (j < b.size())) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else c.push_back(a[i]),i++,j++;
    }
    return c;
}
