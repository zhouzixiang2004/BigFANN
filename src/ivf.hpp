#pragma once
#include <omp.h>
#include "common.hpp"
#include "graph.hpp"
#include "utils.hpp"
#include "distance.hpp"
#include "timer.hpp"
#include <functional>

// everything currently only works for euclidean distance
template <typename T>
void kmeans_cluster(vector_dataset<T> &data, const vector<vid_t> &group,
    int nclusters, vector_dataset<T> &centroids, vector<int> &membership) {
    
    int dim = data.dim;
    assert(centroids.num == 0);
    centroids.num = nclusters;
    centroids.dim = data.dim;
    centroids.data = new T[(int64_t) nclusters*dim];
    membership.resize(group.size());
    assert(size_t(nclusters) <= group.size());

    // initialize centroids at first nclusters points
    for (int i = 0; i < nclusters; i++) {
        #pragma omp simd
        for (int j = 0; j < dim; j++) centroids[i][j] = data[group[i]][j];
    }

    int nthreads = 0;
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    cout << "kmeans " << group.size() << ", " << nclusters << " clusters, " << nthreads << " threads" << endl;

    vector<int> new_centers_len(nclusters);
    vector<vector<float> > new_centers(nclusters,vector<float>(dim));
    vector<vector<int> > partial_new_centers_len(nthreads,vector<int>(nclusters));
    vector<vector<vector<float> > > partial_new_centers(nthreads,
        vector<vector<float> >(nclusters,vector<float>(dim)));
    
    // start clustering
    const int max_iter = 500;
    const int threshold = 0;
    for (int iter = 0; iter < max_iter; iter ++) {
        if (group.size() > 1e5) cout << "iter: " << iter << endl;
        size_t delta = 0;
        float sumdist = 0;
        #pragma omp parallel for schedule(static) reduction(+:delta,sumdist)
        for (size_t pt = 0; pt < group.size(); pt++) {
            int tid = omp_get_thread_num();
            // find the closest cluster center to point pt
            int index = 0;
            float min_dist = FLT_MAX;
            for (int j = 0; j < nclusters; j++) {
                auto dist = compute_distance_squared(dim,data[group[pt]],centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    index = j;
                }
            }
            sumdist += min_dist;
            if (membership[pt] != index) delta += 1;
            membership[pt] = index;
            partial_new_centers_len[tid][index]++;				
            for (int j = 0; j < dim; j++)
                partial_new_centers[tid][index][j] += data[group[pt]][j];
        }

        // let the main thread perform the array reduction
        for (int i = 0; i < nclusters; i++) {
            for (int j = 0; j < nthreads; j++) {
                new_centers_len[i] += partial_new_centers_len[j][i];
                partial_new_centers_len[j][i] = 0;
                for (int k = 0; k < dim; k++) {
                    new_centers[i][k] += partial_new_centers[j][i][k];
                    partial_new_centers[j][i][k] = 0;
                }
            }
        }

        // replace old cluster centers with new_centers
        for (int i = 0; i < nclusters; i++) {
            for (int j = 0; j < dim; j++) {
                if (new_centers_len[i] > 0)
                    centroids[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0;
            }
            //cout << new_centers_len[i] << " ";
            new_centers_len[i] = 0;
        }
        //cout << endl;
        if (delta <= threshold) {
            cout << iter << "," << delta << "," << sumdist << endl;
            break;
        }
    }
}

#include "query_log.hpp"
template <typename T>
class IVF_index {
public:
    vector_dataset<T> centroids;
    vector<vector<int> > ivf;
    int NUM_CLUSTERS = 2;
    int PF_DIST = 4;

    IVF_index(const vector<int> &group,const vector<int> &membership,
        const vector_dataset<T> &_centroids) {
        
        centroids = _centroids;
        ivf.resize(centroids.num);
        for (int i = 0; i < group.size(); i++) ivf[membership[i]].push_back(group[i]);
    }
    IVF_index(const vector<int> &group,const string &filename,bool parlay = false,int dim = 0) {
        if (parlay) {
            std::ifstream in(filename,std::ios::binary);
            in.read((char*) &centroids.num,sizeof(size_t));
            centroids.dim = dim;
            free(centroids.data);
            centroids.data = (T*)malloc((int64_t) centroids.num*dim*sizeof(T));
            in.read((char*) centroids.data,(int64_t) centroids.num*dim*sizeof(T));
            ivf.resize(centroids.num);
            vector<int> indptr(centroids.num+1),indices(group.size());
            in.read((char*) indptr.data(),(centroids.num+1)*sizeof(int));
            in.read((char*) indices.data(),group.size()*sizeof(int));
            for (size_t i = 0; i < centroids.num; i++) {
                for (int j = indptr[i]; j < indptr[i+1]; j++) ivf[i].push_back(indices[j]);
            }
            in.close();
        }
        else {
            vector<int> membership(group.size());
            std::ifstream in(filename + "-mem",std::ios::binary);
            in.read((char*) membership.data(),4*membership.size());
            in.close();
            centroids = vector_dataset<T>((filename + "-cen").c_str());
            ivf.resize(centroids.num);
            for (size_t i = 0; i < group.size(); i++) ivf[membership[i]].push_back(group[i]);
        }
    }

    vector<int> sorted_near(vector_dataset<T> &data,T *query,int target = 15000) {
        vector<pair<float,int> > vv(ivf.size());
        for (int i = 0; i < ivf.size(); i++) {
            vv[i] = make_pair(compute_distance_squared(data.dim,centroids[i],query),i);
        }
        sort(vv.begin(),vv.end());
        vector<int> ret;
        for (int i = 0; i < vv.size(); i++) {
            ret.insert(ret.end(),ivf[vv[i].second].begin(),ivf[vv[i].second].end());
            if (ret.size() > target) break;
        }
        sort(ret.begin(),ret.end());
        return ret;
    }

    vector<int> sorted_near_filter(vector_dataset<T> &data,T *query,
        function<bool(int)> filter,int target = 10) {
        
        //int l = data.dim*sizeof(T)/64;
        vector<pair<float,int> > vv(ivf.size());
        for (size_t i = 0; i < ivf.size(); i++) {
            vv[i] = make_pair(compute_distance_squared(data.dim,centroids[i],query),i);
            // This access pattern is predictable
            #ifdef USE_PF_VEC
            //if (i+PF_DIST < ivf.size()) {
            //    char *a = (char*) centroids[i+PF_DIST];
            //    for (int j = 0; j < l; j++) __builtin_prefetch(a+64*j);
            //}
            #endif
        }
        partial_sort(vv.begin(),vv.begin()+min(NUM_CLUSTERS,(int) vv.size()),vv.end());
        vector<int> ret;
        for (int i = 0; i < min(NUM_CLUSTERS,(int) vv.size()); i++) {
            for (int x: ivf[vv[i].second]) {
                if (filter(x)) ret.push_back(x);
            }
            if (ret.size() > target) break;
        }
        return ret;
    }

    // returns (
    //    the total number of distance computations made,
    //    the number of candidates that passed the filter,
    //    the number of clusters checked
    // )
    tuple<int,int,int> process_filter_query(vector_dataset<T> &data,vector_dataset<int> &results,
        int query_id,T *query,function<bool(int)> filter,int K) {
        
        Timer t_dist;
        t_dist.Start();

        int l = data.dim*sizeof(T)/64;
        vector<pair<float,int> > vv(ivf.size());
        for (size_t i = 0; i < ivf.size(); i++) {
            vv[i] = make_pair(compute_distance_squared(data.dim,centroids[i],query),i);
        }
        t_dist.Stop();
        query_log[query_id].distance_time += t_dist.Seconds();
        sort(vv.begin(),vv.end());

        int pass_filter = 0,total = 0;
        int clusters = 0;
        vector<pair<float,int> > dists(K,make_pair(2e9,-1));
        double filter_time = 0;
        Timer t_filter;
        for (size_t i = 0; i < vv.size(); i++) {
            vector<int> cand;
            t_filter.Start();
            for (int x: ivf[vv[i].second]) {
                if (filter(x)) cand.push_back(x);
            }
            t_filter.Stop();
            filter_time += t_filter.Seconds();
            pass_filter += cand.size();
            total += ivf[vv[i].second].size();
            clusters++;

            t_dist.Start();
            for (size_t j = 0; j < cand.size(); j++) {
                int v = cand[j];
                float d = compute_distance_squared(data.dim,data[v],query);
                if (d < dists.back().first) {
                    dists.pop_back();
                    dists.insert(lower_bound(dists.begin(),dists.end(),make_pair(d,v)),make_pair(d,v));
                }
        #ifdef USE_PF_VEC
                if (j+PF_DIST < cand.size()) {
                    char *a = (char*) data[cand[j+PF_DIST]];
                    for (int k = 0; k < l; k++) __builtin_prefetch(a+64*k);
                }
        #endif
            }
            t_dist.Stop();
            query_log[query_id].distance_time += t_dist.Seconds();
            if (clusters >= NUM_CLUSTERS) break;
        }
        query_log[query_id].filter_time += filter_time;
        for (int i = 0; i < K; i++) results[query_id][i] = dists[i].second;
        return make_tuple(ivf.size()+pass_filter,pass_filter,clusters);
    }
};
