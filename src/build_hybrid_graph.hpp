#include <omp.h>
#include "graph.hpp"
#include "utils.hpp"
#include "timer.hpp"
#include "distance.hpp"
#include "common.hpp"
#include "pqueue.hpp"
#include "hash_table.hpp"
#include "params.hpp"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

// incremental algorithm
template <typename T>
void build_hybrid_graph(vector_dataset<T> &data_vectors,
    const sparse_bitmatrix &filters, const sparse_bitmatrix &filterst,
    vector<Graph*> &_single, Graph *&_shared, vector<vid_t> &init_vertex,
    const BuildParams &P) {

    Timer t_all;
    t_all.Start();
    size_t N = data_vectors.num;
    int dim = data_vectors.dim;
    double alpha = P.ALPHA;
    int L = P.BUILD_BEAM_WIDTH;
    auto distance = [&](int i,int j) {
        // if (data_vectors.INNER_PRODUCT) return compute_ip_distance(dim, data_vectors[i], data_vectors[j]);
        return compute_distance_squared(dim, data_vectors[i], data_vectors[j]);
    };
    auto euclidean_distance = [&](int i,int j) {
        return compute_distance_squared(dim, data_vectors[i], data_vectors[j]);
    };
    if (data_vectors.INNER_PRODUCT) {
        cout << "inner product distance" << endl;
        assert(0 && "Inner product distance is not supported"); // no support
    }
    
    // gather all the big labels
    vector<size_t> big;
    vector<size_t> ibig(filterst.rows,-1);
    for (size_t i = 0; i < filterst.rows; i++) {
        if (filterst[i].size() > (size_t)P.BIG_CUTOFF) ibig[i] = big.size(),big.push_back(i);
    }
    cout << "big: " << big.size() << endl;

    // build hash_subgroup data structure for all big labels
    vector<hash_subgroup<vid_t>*> lookup(big.size());
    for (size_t i = 0; i < big.size(); i++) {
        vector<vid_t> group;
        for (auto v: filterst[big[i]]) group.push_back(v);
        lookup[i] = new hash_subgroup(group);
    }

    // decide degrees of single-label graphs
    vector<size_t> D(big.size());
    for (size_t i = 0; i < big.size(); i++) {
        D[i] = (size_t)(-1);
        for (auto [ub,d]: P.GRAPH_DEGREE) {
            if ((D[i] == (size_t)(-1)) && (filterst[big[i]].size() <= size_t(ub))) D[i] = d;
        }
        assert(D[i] != (size_t)(-1));
    }
    size_t shared_D = P.SHARED_DEGREE;

    size_t batch_bound = 0.01*N;
    size_t start = 1;
    // single[label id][in-group node id][neighbor] -> (distance,in-group node id)
    vector<vector<vector<tuple<float,vid_t>>>> single(big.size());
    for (size_t i = 0; i < big.size(); i++) single[i].resize(filterst[big[i]].size());
    // shared[node id][neighbor] -> (distance,overlap,node id)
    vector<vector<tuple<float,uint8_t,vid_t>>> shared(N);
    auto prune_shared = [&](vid_t u) {
        if (shared[u].size() > shared_D) {
            sort(shared[u].begin(),shared[u].end());
            vector<tuple<float,uint8_t,vid_t> > keep;
            for (double a: {1.0,alpha}) {
                for (auto [duv,overv,v]: shared[u]) {
                    if (keep.size() == shared_D) break;
                    bool bad = false;
                    for (auto [duw,overw,w]: keep) {
                        if (a*euclidean_distance(w,v) <= duv) {
                            bad = true;
                            break;
                        }
                    }
                    if (!bad) keep.emplace_back(duv,overv,v);
                }
            }
            shared[u] = keep;
            shared[u].shrink_to_fit();
        }
    };
    auto prune_single = [&](size_t group_id,vid_t id_u) {
        if (single[group_id][id_u].size() > D[group_id]) {
            vid_t u = filterst[big[group_id]][id_u];
            auto &cand = single[group_id][id_u];
            sort(cand.begin(),cand.end());
            vector<tuple<float,vid_t> > relevant_shared,keep;
            // gather shared edges relevant to this group
            for (auto [duv,overv,v]: shared[u]) {
                // if total overlap, don't need to check
                if ((overv == filters[u].size()) || filters[v].contains(big[group_id])) {
                    relevant_shared.emplace_back(duv,v);
                }
            }
            // prune
            for (double a: {1.0,alpha}) {
                for (auto [duv,id_v]: cand) {
                    if (keep.size() == D[group_id]) break;
                    vid_t v = filterst[big[group_id]][id_v];
                    bool bad = false;
                    for (auto [duw,w]: relevant_shared) {
                        if (a*euclidean_distance(w,v) <= duv) {
                            bad = true;
                            break;
                        }
                    }
                    if (!bad) {
                        for (auto [duw,id_w]: keep) {
                            vid_t w = filterst[big[group_id]][id_w];
                            if (a*euclidean_distance(w,v) <= duv) {
                                bad = true;
                                break;
                            }
                        }
                    }
                    if (!bad) keep.emplace_back(duv,id_v);
                }
            }
            cand = keep;
            cand.shrink_to_fit();
        }
    };

    // insert in random order
    mt19937 gen(1234);
    vector<size_t> order(N);
    for (size_t i = 0; i < N; i++) order[i] = i;
    shuffle(order.begin(),order.end(),gen);
    // get first vertex of each label
    init_vertex.resize(big.size(),-1);
    for (size_t i = 0; i < N; i++) {
        vid_t ins = order[i];
        for (auto f: filters[ins]) {
            if ((ibig[f] != (size_t)(-1)) && (init_vertex[ibig[f]] == -1)) {
                init_vertex[ibig[f]] = ins;
            }
        }
    }

    while (start < N) {
        Timer t_iter;
        t_iter.Start();
        size_t end = min(min(2*start,start+batch_bound),N);
        cout << start << " to " << end << endl;

        #pragma omp parallel for
        for (size_t i = start; i < end; i++) {
            vid_t ins = order[i];
            for (auto f: filters[ins]) {
                if (ibig[f] == (size_t)(-1)) continue;
                size_t group_id = ibig[f];
                if (init_vertex[group_id] == ins) continue;
                //cout << "inserting " << ins << ", group " << f << ", init node " << init_vertex[group_id] << endl;

                // filtered beam search with f
                hash_filter<vid_t> is_visited(min((size_t) L*L,(size_t) filterst[f].size()));
                new_pqueue_t<vid_t> S(L);
                S.push(init_vertex[group_id],distance(init_vertex[group_id],ins));
                is_visited.add(init_vertex[group_id]);

                auto add_edge = [&](vid_t u,vid_t v,float dist) {
                    // add candidate edge from u to v
                    uint8_t overlap = (uint8_t)min(filters[u].get_intersect_num(filters[v]),255U);
                    if (overlap >= P.MIN_SHARED) shared[u].emplace_back(dist,overlap,v);
                    vid_t id_u = lookup[group_id]->pos(u);
                    vid_t id_v = lookup[group_id]->pos(v);
                    single[group_id][id_u].emplace_back(dist,id_v);
                };

                vector<vid_t> keep;
                while (S.has_unexpanded()) {
                    int idx = S.get_next_index();
                    vid_t u = S[idx];
                    auto dui = S.get_dist(idx);
                    add_edge(ins,u,dui);
                    S.set_front_expanded();

                    keep.clear();
                    vid_t id_u = lookup[group_id]->pos(u);
                    for (auto [duv,id_v]: single[group_id][id_u]) {
                        vid_t v = filterst[f][id_v];
                        if (is_visited.add(v)) {
                            keep.push_back(v);
                            char *a = (char*)data_vectors[v];
                            int l = dim*sizeof(T)/64;
                            for (int j = 0; j < l; j++) __builtin_prefetch(a+64*j);
                        }
                    }
                    for (auto [duv,overv,v]: shared[u]) {
                        if ((overv == filters[u].size()) || filters[v].contains(f)) {
                            if (is_visited.add(v)) {
                                keep.push_back(v);
                                char *a = (char*)data_vectors[v];
                                int l = dim*sizeof(T)/64;
                                for (int j = 0; j < l; j++) __builtin_prefetch(a+64*j);
                            }
                        }
                    }

                    for (auto v: keep) {
                        auto dist = distance(v,ins);
                        if ((S.size() < L) || (dist < S.get_tail_dist())) {
                            S.push(v,dist);
                        }
                    }
                }

                prune_shared(ins);
            }
            for (auto f: filters[ins]) {
                if (ibig[f] == (size_t)(-1)) continue;
                size_t group_id = ibig[f];
                prune_single(group_id,lookup[group_id]->pos(ins));
            }
        }

        t_iter.Stop();
        printf("Time to insert nodes: %f sec\n",t_iter.Seconds());
        fflush(stdout);
        t_iter.Start();

        // reverse edges
        // TODO parallelize this part
        vector<vid_t> to_update_shared;
        vector<pair<size_t,vid_t> > to_update_single;
        for (size_t i = start; i < end; i++) {
            vid_t u = order[i];
            for (auto f: filters[u]) {
                if (ibig[f] == (size_t)(-1)) continue;
                size_t group_id = ibig[f];
                auto id_u = lookup[group_id]->pos(u);

                for (auto [duv,id_v]: single[group_id][id_u]) {
                    single[group_id][id_v].emplace_back(duv,id_u);
                    to_update_single.emplace_back(group_id,id_v);
                }
            }
            for (auto [duv,overv,v]: shared[u]) {
                shared[v].emplace_back(duv,overv,u);
                to_update_shared.push_back(v);
            }
        }
        /*auto to_update_shared_sorted = parlay::sort(to_update_shared);
        auto to_update_single_sorted = parlay::sort(to_update_single);
        parlay::parallel_for(0,to_update_shared_sorted.size(),[&](size_t i) {
            if ((i == 0) || (to_update_shared_sorted[i] != to_update_shared_sorted[i-1]))
                prune_shared(to_update_shared_sorted[i]);
        });
        parlay::parallel_for(0,to_update_single_sorted.size(),[&](size_t i) {
            if ((i == 0) || (to_update_single_sorted[i] != to_update_single_sorted[i-1]))
                prune_single(to_update_single_sorted[i].first,
                             to_update_single_sorted[i].second);
        });*/
        sort(to_update_shared.begin(),to_update_shared.end());
        sort(to_update_single.begin(),to_update_single.end());
        #pragma omp parallel for
        for (size_t i = 0; i < to_update_shared.size(); i++) {
            if ((i == 0) || (to_update_shared[i] != to_update_shared[i-1]))
                prune_shared(to_update_shared[i]);
        }
        #pragma omp parallel for
        for (size_t i = 0; i < to_update_single.size(); i++) {
            if ((i == 0) || (to_update_single[i] != to_update_single[i-1]))
                prune_single(to_update_single[i].first,to_update_single[i].second);
        }

        t_iter.Stop();
        printf("Time to add reverse edges: %f sec\n",t_iter.Seconds());

        start = end;
    }

    // construct the graph
    _shared = new RegularGraph;
    _shared->allocateFrom(N,(int64_t) N*shared_D);
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        auto offset = (int64_t) i*shared_D;
        _shared->fixEndEdge(i,offset + shared_D);
        assert(shared[i].size() <= shared_D);
        for (size_t j = 0; j < shared[i].size(); j++) _shared->constructEdge(offset+j,get<2>(shared[i][j]));
        for (size_t j = shared[i].size(); j < shared_D; j++) _shared->constructEdge(offset+j,0);
    }
    _single.resize(big.size());
    for (size_t group_id = 0; group_id < big.size(); group_id++) {
        _single[group_id] = new RegularGraph;
        auto g = _single[group_id];
        size_t GN = filterst[big[group_id]].size();

        g->allocateFrom(GN,(int64_t) GN*D[group_id]);
        #pragma omp parallel for
        for (size_t i = 0; i < GN; i++) {
            auto offset = (int64_t) i*D[group_id];
            g->fixEndEdge(i,offset + D[group_id]);
            assert(single[group_id][i].size() <= D[group_id]);
            for (size_t j = 0; j < single[group_id][i].size(); j++) g->constructEdge(offset+j,get<1>(single[group_id][i][j]));
            for (size_t j = single[group_id][i].size(); j < D[group_id]; j++) g->constructEdge(offset+j,0);
        }
    }

    t_all.Stop();
    printf("Building hybrid graph using incremental alg took time: %f sec\n", t_all.Seconds());
}