#include "common.hpp"
#include "distance.hpp"
#include "utils.hpp"
#include "params.hpp"
#include "pqueue.hpp"
#include "hash_table.hpp"
#include "parlay/parallel.h"
#include "query_log.hpp"
#include "timer.hpp"
#include "graph.hpp"
#include "ivf.hpp"
#include <map>
using namespace std;

string DATA_DIR = "./data/";
string dataset_name;

template <typename T>
int getKNN(vector_dataset<T> &data,vector_dataset<int> &results,
    int query_id,T *query_data,vidType *candidates,int num,const QueryParams &P) {
    
    int l = data.dim*sizeof(T)/64;
    Timer distance_timer;
    double distance_time = 0;
    vector<pair<float,vidType> > dists(P.K,make_pair(2e9,-1));
    for (int i = 0; i < num; i++) {
        auto v = candidates[i];
        //distance_timer.Start();
        float d = compute_distance_squared(data.dim,data[v],query_data);
        //distance_timer.Stop();
        //distance_time += distance_timer.Seconds();
        if (d < dists.back().first) {
            dists.pop_back();
            dists.insert(lower_bound(dists.begin(),dists.end(),make_pair(d,v)),make_pair(d,v));
        }
#ifdef USE_PF_VEC
        if (i+P.PF_DIST < num) {
            char *a = (char*) data[candidates[i+P.PF_DIST]];
            for (int j = 0; j < l; j++) __builtin_prefetch(a+64*j);
        }
#endif
    }
    for (int j = 0; j < P.K; j++) results[query_id][j] = dists[j].second;
    query_log[query_id].distance_time += distance_time;
    return num;
}

ofstream out;
string index_f;

template<typename T>
void run_hybrid_query(vector_dataset<T> &data, string filters_file,
    string queries_file, string query_filters_file, string gt_file,
    string index_folder, const QueryParams &P,
    RegularGraph *shared, const vector<RegularGraph*> &single) {
    cout << "run_hybrid_query" << endl;
    
    // we take data as an argument so it's not reloaded every time
    // vector_dataset<T> data(data_file.c_str());
    sparse_bitmatrix filters(filters_file.c_str());
    sparse_bitmatrix filterst = transpose(filters);
    vector_dataset<T> queries(queries_file.c_str());
    sparse_bitmatrix query_filters(query_filters_file.c_str());
    vector_dataset<int> gt_all(gt_file.c_str());
    cout << "loaded stuff" << endl;

    vector_dataset<int> results;
    results.num = queries.num;
    results.dim = P.K;
    results.data = new int[queries.num*P.K];
    query_log.resize(queries.num);

    vector<size_t> big;
    vector<size_t> ibig(filterst.rows,-1);
    for (size_t i = 0; i < filterst.rows; i++) {
        if (filterst[i].size() > (size_t)P.BIG_CUTOFF) ibig[i] = big.size(),big.push_back(i);
    }
    cout << "big: " << big.size() << endl;

    // build hash_subgroup data structure for all big labels
    vector<hash_subgroup<vid_t>*> lookup(big.size());
    for (size_t i = 0; i < big.size(); i++) {
        //cout << "built hash subgroup " << i << endl;
        vector<vid_t> group;
        for (auto v: filterst[big[i]]) group.push_back(v);
        lookup[i] = new hash_subgroup(group);
    }

    vector<vid_t> init_vertex;
    ifstream init(index_folder + "/init_vertex.ibin",ios::binary);
    int siz;
    init.read((char*)&siz, 4);
    init_vertex.resize(siz);
    init.read((char*)init_vertex.data(), sizeof(vid_t)*siz);
    cout << "loaded init_vertex" << endl;

    int c = 0;
    vector<vector<int> > pos_precomp(big.size(),vector<int>(big.size()));
    for (size_t i = 0; i < big.size(); i++) {
        for (size_t j = i+1; j < big.size(); j++) pos_precomp[i][j] = c++;
    }
    sparse_bitmatrix precomp((index_folder + "/precomp").c_str(),false);

    vector<IVF_index<T>*> ivf(big.size());
#ifdef USE_PARLAY
    parlay::parallel_for(0,big.size(),[&](size_t i) {
#else
    #pragma omp parallel for
    for (size_t i = 0; i < big.size(); i++) {   
#endif
        vector<vid_t> group;
        for (auto v: filterst[big[i]]) group.push_back(v);
        //if ((i % 100) == 0) cout << "loading ivf " << i << " of " << big.size() << endl;
        string name;
        if (dataset_name.substr(0,8) != "sift100m") {
            name = index_folder + "/ivf-" + to_string(i);
            ivf[i] = new IVF_index<T>(group,name);
        }
        else {
            name = to_string(big[i]) + "_postingList_"+to_string(group.size()/10000)+"_"+"5"+".bin";
            name = DATA_DIR + dataset_name + "/index_cache_32/" + name;
            ivf[i] = new IVF_index<T>(group,name,true,data.dim);
        }

        ivf[i]->PF_DIST = P.PF_DIST;
        ivf[i]->NUM_CLUSTERS = P.NUM_CLUSTERS;
    }
#ifdef USE_PARLAY
    );
#endif

    // assume the graphs are pre-loaded
    /*RegularGraph *shared = new RegularGraph(index_folder + "/shared");
    vector<RegularGraph*> single(big.size());
    for (size_t i = 0; i < big.size(); i++) {
        single[i] = new RegularGraph(index_folder + "/single-" + to_string(i));
    }
    cout << "loaded graphs" << endl;*/

    vector<vector<uint64_t> > bitsets(filterst.rows);
    for (size_t i = 0; i < filterst.rows; i++) {
        if (filterst[i].size() > (size_t)P.BITSET_CUTOFF) {
            bitsets[i].resize((data.num+63)/64);
            for (auto v: filterst[i]) bitsets[i][v >> 6] |= (1ULL << (v & 63));
        }
    }
    cout << "finished building bitsets" << endl;
    auto contains = [&](int f,vid_t v) -> bool {
        if (filterst[f].size() > (size_t)P.BITSET_CUTOFF)
            return bitsets[f][v >> 6] & (1ULL << (v & 63));
        else
            return filters[v].contains(f);
    };

    int dim = queries.dim;
    int L = P.QUERY_BEAM_WIDTH;
    cout << "query beam width = " << L << endl;
    cout << "num clusters = " << P.NUM_CLUSTERS << endl;
    int K = P.K;

    Timer all_timer;
    all_timer.Start();

    vector<size_t> order(queries.num);
#ifdef USE_PARLAY
    parlay::parallel_for(0,queries.num,[&](size_t i) {
#else
    #pragma omp parallel for
    for (size_t i = 0; i < queries.num; i++) {
#endif
        order[i] = i;
        if (query_filters[i].size() == 2) {
            int a = query_filters[i][0];
            int b = query_filters[i][1];
            if (make_pair(filterst[a].size(),a) > make_pair(filterst[b].size(),b))
                swap(query_filters[i][0],query_filters[i][1]);
        }
    }
#ifdef USE_PARLAY
    );
#endif
    sort(order.begin(),order.end(),[&](int x,int y) {
        int large_filter_x = filterst[query_filters[x][query_filters[x].size()-1]].size();
        int large_filter_y = filterst[query_filters[y][query_filters[y].size()-1]].size();
        if (large_filter_x == large_filter_y) {
            if (query_filters[x].size() == query_filters[y].size()) {
                int small_filter_x = filterst[query_filters[x][0]].size();
                int small_filter_y = filterst[query_filters[y][0]].size();
                return small_filter_x > small_filter_y;
            }
            else return query_filters[x].size() > query_filters[y].size();
        }
        else return large_filter_x > large_filter_y;
    });

#ifdef USE_PARLAY
    parlay::parallel_for(0,queries.num,[&](size_t i) {
#else
    #pragma omp parallel for schedule(dynamic,1)
    for (size_t i = 0; i < queries.num; i++) {
#endif
        size_t query_id = order[i];
        Timer t;
        t.Start();

        if (query_filters[query_id].size() == 2) {
            // double-filter
            auto x = query_id;
            auto query_data = queries[x];
            assert(query_filters[x].size() == 2);
            int a = query_filters[x][0];
            int b = query_filters[x][1];
            assert(make_pair(filterst[a].size(),a) < make_pair(filterst[b].size(),b));
            
            if (filterst[a].size() <= (size_t)P.BIG_CUTOFF) {
                // one is <= BIG_CUTOFF, brute2
                int this_count_dc = 0;

                if (filterst[b].size() > (size_t)P.BITSET_CUTOFF) {
                    // the larger filter has a bitset precomputed
                    vector<vidType> c;
                    Timer t_filter;
                    t_filter.Start();
                    for (auto v: filterst[a]) {
                        // v is a vector id that matches the first filter
                        if (bitsets[b][v >> 6] & (1ULL << (v & 63)))
                            c.push_back(v);
                    }
                    t_filter.Stop();
                    query_log[x].filter_time = t_filter.Seconds();
                    this_count_dc += getKNN(data,results,x,query_data,c.data(),c.size(),P);
                }
                else {
                    // otherwise, compute join
                    vector<vidType> c(filterst[a].size());
                    Timer t_filter;
                    t_filter.Start();
                    int s = filterst[a].intersect_buf(c.data(),filterst[b]);
                    t_filter.Stop();
                    query_log[x].filter_time = t_filter.Seconds();
                    this_count_dc += getKNN(data,results,x,query_data,c.data(),s,P);
                }

                t.Stop();
                query_log[x].id = x;
                query_log[x].distance_comps = this_count_dc;
                query_log[x].time = t.Seconds();
                query_log[x].cas = "brute2-";
                if (filterst[b].size() <= 60000)
                    query_log[x].cas += "smallxsmall";
                else 
                    query_log[x].cas += "tinyxlarge";
            }
            else {
                // both large
                int pos1 = ibig[a];
                int pos2 = ibig[b];
                assert(pos1 != -1);
                assert(pos2 != -1);
                if (pos1 > pos2) swap(pos1,pos2);
                int p = pos_precomp[pos1][pos2];

                if (precomp[p].size() <= (size_t)P.PRECOMP_CUTOFF) {
                    // precomputed intersection is small
                    getKNN(data,results,x,query_data,precomp.get_begin(p),
                        precomp.get_size(p),P);

                    t.Stop();
                    query_log[x].id = x;
                    query_log[x].distance_comps = precomp[p].size();
                    query_log[x].time = t.Seconds();
                    query_log[x].cas = "precomp-";
                    if (filterst[b].size() <= 60000)
                        query_log[x].cas += "smallxsmall";
                    else if (filterst[a].size() <= 60000)
                        query_log[x].cas += "tinyxlarge";
                    else 
                        query_log[x].cas += "largexlarge";
                }
                else {
                    // intersection is large, use ivf
                    bool has_bitset = (filterst[b].size() > (size_t)P.BITSET_CUTOFF);
                    int checks = 0;
                    auto check = [&](int v) -> bool {
                        checks++;
                        if (has_bitset) {
                            if (!(bitsets[b][v >> 6] & (1ULL << (v & 63)))) {
                                return false;
                            }
                            else {
                                return true;
                            }
                        }
                        else {
                            if (!filters[v].contains(b)) {
                                return false;
                            }
                            else {
                                return true;
                            }
                        }
                    };

                    //vector<int> c = ivf[pos[a]]->sorted_near_filter(data,queries[x],check);
                    //getKNN(data,results,x,query_data,c.data(),c.size(),P);
                    auto metadata = ivf[ibig[a]]->process_filter_query(data,results,x,queries[x],check,P.K);

                    t.Stop();
                    query_log[x].id = x;
                    //query_log[x].distance_comps = ivf[pos[a]]->ivf.size()+c.size();
                    query_log[x].distance_comps = get<0>(metadata);
                    query_log[x].time = t.Seconds();
                    query_log[x].cas = "ivf-";
                    query_log[x].checks = checks;
                    //query_log[x].pass_filter = c.size();
                    query_log[x].pass_filter = get<1>(metadata);
                    query_log[x].clusters = get<2>(metadata);
                    if (filterst[b].size() <= 60000)
                        query_log[x].cas += "smallxsmall";
                    else if (filterst[a].size() <= 60000)
                        query_log[x].cas += "tinyxlarge";
                    else 
                        query_log[x].cas += "largexlarge";
                }
            }
        }
        else {
            // single-filter
            assert(query_filters[query_id].size() == 1);
            int a = query_filters[query_id][0];
            auto query_data = queries[query_id];

            if (filterst[a].size() <= (size_t)P.BIG_CUTOFF) {
                // <= BIG_CUTOFF, brute
                getKNN(data,results,query_id,query_data,filterst.get_begin(a),
                    filterst.get_size(a),P);

                t.Stop();
                query_log[query_id].id = query_id;
                query_log[query_id].distance_comps = filterst[a].size();
                query_log[query_id].time = t.Seconds();
                query_log[query_id].cas = "brute";
            }
            else {
                assert(ibig[a] != (size_t)(-1));
                int group_id = ibig[a];
                // store nodes in the in-group id space
                hash_filter<vid_t> is_visited(min(L*L,(int)filterst[a].size()));
                new_pqueue_t<vid_t> S(L); // priority queue

                // only Euclidean distance for now
                auto distance_to = [&](vid_t v) {
                    return compute_distance_squared(dim, data[filterst[a][v]], query_data);
                };

                int this_count_dc = 1;
                vid_t u = lookup[group_id]->pos(init_vertex[group_id]);
                S.push(u,distance_to(u));
                is_visited.add(u);

                // start search until no more un-expanded nodes in the queue
                int iter = 0, num_hops = 0, num_push = 0;
                vector<vid_t> keep;
                query_log[query_id].filter_time = 0;
                Timer tt;
                while (S.has_unexpanded()) {
                    ++iter;
                    int idx = S.get_next_index();
                    int u = S[idx];
                    single[group_id]->fastprefetch(u,_MM_HINT_T0);
                    int glob_u = filterst[a][u];
                    shared->fastprefetch(glob_u,_MM_HINT_T0);
                    S.set_front_expanded();
            
                    keep.clear();
                    for (auto v: single[group_id]->fastN(u)) {
                        if (is_visited.add(v)) {
                            keep.push_back(v);
                            PREFETCH_VECTOR(dim, data[filterst[a][v]]);
                        }
                    }
                    tt.Start();
                    for (auto glob_v: shared->fastN(glob_u)) {
                        // here, could speed up maximal edges like UNG?
                        if (contains(a,glob_v)) {
                            auto v = lookup[group_id]->pos(glob_v);
                            if (is_visited.add(v)) {
                                keep.push_back(v);
                                PREFETCH_VECTOR(dim, data[glob_v]);
                            }
                        }
                    }
                    tt.Stop();
                    query_log[query_id].filter_time += tt.Seconds();

                    for (auto v: keep) {
                        //distance_timer.Start();
                        auto dist = distance_to(v);
                        //distance_timer.Stop();
                        //distance_time += distance_timer.Seconds();
                        this_count_dc++;
                        //if ((S.size() > K) && (dist > S.get_dist(K)*1.35)) continue;
                        if ((S.size() < L) || (dist < S.get_tail_dist())) {
                            S.push(v,dist);
                            num_push++;
                        }
                    }
            
                    ++num_hops;
                }

                for (int i = 0; i < K; ++ i) {
                    results[query_id][i] = filterst[a][S[i]];
                }
                t.Stop();
                double runtime = t.Seconds();

                query_log[query_id].id = query_id;
                query_log[query_id].distance_comps = this_count_dc;
                query_log[query_id].time = runtime;
                query_log[query_id].cas = "hybridgraph";
                query_log[query_id].iter = iter;
                query_log[query_id].hops = num_hops;
                query_log[query_id].push = num_push;
            }
        }
    }
#ifdef USE_PARLAY
    );
#endif

    // overall performance
    all_timer.Stop();
    double runtime = all_timer.Seconds();
    auto throughput = double(queries.num) / runtime;
    auto recall = compute_avg_recall_1D(results,gt_all);
    cout << "runtime: " << runtime << endl;
    cout << "recall: " << recall << endl;
    cout << "QPS: " << throughput << endl;

    // each case recall, throughput
    map<string,int> num_correct,num_all;
    map<string,int64_t> num_dist_comps;
    map<string,double> num_time,num_distance_time;
    double total_distance_time = 0;
    for (size_t q_i = 0; q_i < queries.num; q_i++) {
        int correct = 0;
        for (int top_i = 0; top_i < P.K; ++top_i) {
            auto true_id = gt_all[q_i][top_i];
            for (int n_i = 0; n_i < P.K; ++n_i) {
                if (results[q_i][n_i] == true_id) {
                    correct ++;
                    break;
                }
            }
        }
        for (int j = 0; j < P.K; j++) {
            if (!filters[results[q_i][j]].contains(query_filters[q_i][0])) {
                if (q_i == 0) cout << q_i << "is bad " << results[q_i][j] << endl;
            }
        }

        num_correct[query_log[q_i].cas] += correct;
        num_all[query_log[q_i].cas]++;
        num_time[query_log[q_i].cas] += query_log[q_i].time;
        num_distance_time[query_log[q_i].cas] += query_log[q_i].distance_time;
        total_distance_time += query_log[q_i].distance_time;
        num_dist_comps[query_log[q_i].cas] += query_log[q_i].distance_comps;

        query_log[q_i].recall = correct;
        query_log[q_i].filter_count_a = filterst[query_filters[q_i][0]].size();
        query_log[q_i].filter_count_b = (query_filters[q_i].size() == 1) ? 0:filterst[query_filters[q_i][1]].size();
    }
    int64_t total_dist_comps = 0;
    cout << "printing log to " << ("./logs/" + dataset_name + ".log") << endl;
    ofstream outlog("./logs/" + dataset_name + ".log");
    for (auto [cas,a]: num_all) {
        outlog << cas << " " << (double) num_correct[cas]/P.K/a << " recall, ";
        outlog << num_time[cas] << " time (" << (double) a/num_time[cas] << " QPS), ";
        outlog << num_distance_time[cas] << " distance time, ";
        outlog << num_dist_comps[cas] << " dist comps, ";
        outlog << a << " total" << endl;
        total_dist_comps += num_dist_comps[cas];
    }
    cout << "total dist comps: " << total_dist_comps << endl;
    cout << "average dist comps: " << (double) total_dist_comps/queries.num << endl;
    cout << "total distance time: " << total_distance_time << endl;
    cout << "----------------------------------------------" << endl;

    // print to log csv
    out << index_f << ","
        << P.QUERY_BEAM_WIDTH << ","
        << P.NUM_CLUSTERS << ","
        << runtime << ","
        << recall << ","
        << throughput << ","
        << (double) total_dist_comps/queries.num << endl;

    // detailed per-query logs
    ofstream querylog("./logs/" + dataset_name + "_log.csv");
    querylog << ",comparisons,time,distance_time,filter_count_a,filter_count_b,recall,case,iter,hops,push,checks,pass_filter,clusters,filter_time" << endl;
    for (size_t i = 0; i < queries.num; i++) {
        querylog << query_log[i].id << ",";
        querylog << query_log[i].distance_comps << ",";
        querylog << query_log[i].time << ",";
        querylog << query_log[i].distance_time << ",";
        querylog << query_log[i].filter_count_a << ",";
        querylog << query_log[i].filter_count_b << ",";
        querylog << query_log[i].recall << ",";
        querylog << query_log[i].cas << ",";
        querylog << query_log[i].iter << ",";
        querylog << query_log[i].hops << ",";
        querylog << query_log[i].push << ",";
        querylog << query_log[i].checks << ",";
        querylog << query_log[i].pass_filter << ",";
        querylog << query_log[i].clusters << ",";
        querylog << query_log[i].filter_time << endl;
    }

    for (size_t i = 0; i < big.size(); i++) {
        delete lookup[i];
    }
}

void usage(int argc, char *argv[]) {
    if (argc < 3) {
      fprintf(stderr, "Usage: %s "
              "<dataset> "
              "<folder> <beam_widths or num_clusters>\n"
              , argv[0]);
      exit(EXIT_FAILURE);
    }
  }

int main(int argc, char *argv[]) {
    usage(argc, argv);
    dataset_name = argv[1];
    string index_folder = argv[2];
    index_folder = DATA_DIR + dataset_name + "/" + index_folder;

    string query_type = "single";

    string vector_type;
    string data_file, filters_file, queries_file, query_filters_file, gt_file;
    QueryParams P;
    if (dataset_name == "yfcc100M") {
        data_file = "base.10M.u8bin.crop_nb_10000000";
        filters_file = "base.metadata.10M.spmat";
        queries_file = "query.public.100K.u8bin";
        query_filters_file = "query.metadata.public.100K.spmat";
        gt_file = "GT.public.ibin";
        vector_type = "uint8";
        P.BIG_CUTOFF = 5000;
        P.BITSET_CUTOFF = 10000;
        P.PRECOMP_CUTOFF = 2000;
        P.QUERY_BEAM_WIDTH = 85;
        P.NUM_CLUSTERS = 10;
    }
    else if (dataset_name == "yfcc100M-single") {
        dataset_name = "yfcc100M";
        index_folder = DATA_DIR + dataset_name + "/" + argv[2];
        data_file = "base.10M.u8bin.crop_nb_10000000";
        filters_file = "base.metadata.10M.spmat";
        queries_file = "single_filter_query.public.100K.u8bin";
        query_filters_file = "single_filter_query.metadata.public.100K.spmat";
        gt_file = "single_GT.public.ibin";
        vector_type = "uint8";
        P.BIG_CUTOFF = 5000;
        P.BITSET_CUTOFF = 10000;
        P.PRECOMP_CUTOFF = 2000;
        P.QUERY_BEAM_WIDTH = 85;
        P.NUM_CLUSTERS = 2;
    }
    else {
        data_file = dataset_name + "_base.fbin";
        filters_file = "base.metadata.spmat";
        queries_file = dataset_name + "_query.fbin";
        query_filters_file = "single_query.metadata.spmat";
        gt_file = "single_GT.ibin";
        vector_type = "float";
        if (dataset_name == "sift") {
            P.BIG_CUTOFF = 0;
            P.BITSET_CUTOFF = 0;
            P.PRECOMP_CUTOFF = 0;
            P.QUERY_BEAM_WIDTH = 0;
            P.NUM_CLUSTERS = 0;
        }
        else if (dataset_name == "sift-double") {
            dataset_name = "sift";
            index_folder = DATA_DIR + dataset_name + "/" + argv[2];
            data_file = dataset_name + "_base.fbin";
            queries_file = dataset_name + "_query.fbin";
            query_filters_file = "double_query.metadata.spmat";
            gt_file = "double_GT.ibin";
            query_type = "double";
            P.BIG_CUTOFF = 0;
            P.BITSET_CUTOFF = 0;
            P.PRECOMP_CUTOFF = 0;
            P.QUERY_BEAM_WIDTH = 0;
            P.NUM_CLUSTERS = 0;
        }
        else if (dataset_name == "sift100m") {
            data_file = dataset_name + "_base.u8bin";
            queries_file = dataset_name + "_query.u8bin";
            vector_type = "uint8";
            P.BIG_CUTOFF = 0;
            P.BITSET_CUTOFF = 0;
            P.PRECOMP_CUTOFF = 0;
            P.QUERY_BEAM_WIDTH = 0;
            P.NUM_CLUSTERS = 0;
        }
        else if (dataset_name == "sift100m-double") {
            dataset_name = "sift100m";
            index_folder = DATA_DIR + dataset_name + "/" + argv[2];
            data_file = dataset_name + "_base.u8bin";
            queries_file = dataset_name + "_query.u8bin";
            vector_type = "uint8";
            query_filters_file = "double_query.metadata.spmat";
            gt_file = "double_GT.ibin";
            P.BIG_CUTOFF = 0;
            P.BITSET_CUTOFF = 0;
            P.PRECOMP_CUTOFF = 0;
            P.QUERY_BEAM_WIDTH = 0;
            P.NUM_CLUSTERS = 0;
        }
        else if (dataset_name == "gist") {
            P.BIG_CUTOFF = 0;
            P.BITSET_CUTOFF = 0;
            P.PRECOMP_CUTOFF = 0;
            P.QUERY_BEAM_WIDTH = 0;
            P.NUM_CLUSTERS = 0;
        }
        else if (dataset_name == "paper") {
            P.BIG_CUTOFF = 0;
            P.BITSET_CUTOFF = 0;
            P.PRECOMP_CUTOFF = 0;
            P.QUERY_BEAM_WIDTH = 0;
            P.NUM_CLUSTERS = 0;
        }
        else if (dataset_name == "arxiv") {
            P.BIG_CUTOFF = 0;
            P.BITSET_CUTOFF = 0;
            P.PRECOMP_CUTOFF = 0;
            P.QUERY_BEAM_WIDTH = 0;
            P.NUM_CLUSTERS = 0;
        }
        else fprintf(stderr,"Unknown dataset"),exit(EXIT_FAILURE);
    }
    data_file = DATA_DIR + dataset_name + "/" + data_file;
    filters_file = DATA_DIR + dataset_name + "/" + filters_file;
    queries_file = DATA_DIR + dataset_name + "/" + queries_file;
    query_filters_file = DATA_DIR + dataset_name + "/" + query_filters_file;
    gt_file = DATA_DIR + dataset_name + "/" + gt_file;

    index_f = string(argv[2]);
    out = ofstream("logs/" + dataset_name + "_hybrid.csv", ios::app);
    if (out.tellp() == 0) {
        // File is empty, print the header
        out << "index,beam_width,num_clusters,runtime,recall,qps,avg_dist_comps" << endl;
    }

    sparse_bitmatrix filters(filters_file.c_str());
    sparse_bitmatrix filterst = transpose(filters);
    vector<size_t> big;
    for (size_t i = 0; i < filterst.rows; i++) {
        if (filterst[i].size() > (size_t)P.BIG_CUTOFF) big.push_back(i);
    }
    RegularGraph *shared = new RegularGraph(index_folder + "/shared");
    vector<RegularGraph*> single(big.size());
    for (size_t i = 0; i < big.size(); i++) {
        single[i] = new RegularGraph(index_folder + "/single-" + to_string(i));
    }
    cout << "loaded graphs" << endl;

    if (vector_type == "uint8") {
        cout << "loading vector data" << endl;
        vector_dataset<uint8_t> data(data_file.c_str());
        cout << "loaded vector data" << endl;
        for (int i = 3; i < argc; i++) {
            if (query_type == "double") P.NUM_CLUSTERS = stoi(argv[i]);
            else P.QUERY_BEAM_WIDTH = stoi(argv[i]);
            run_hybrid_query<uint8_t>(data,filters_file,queries_file,
                query_filters_file,gt_file,index_folder,P,
                shared,single);
        }
    }
    else if (vector_type == "float") {
        vector_dataset<float> data(data_file.c_str());
        for (int i = 3; i < argc; i++) {
            if (query_type == "double") P.NUM_CLUSTERS = stoi(argv[i]);
            else P.QUERY_BEAM_WIDTH = stoi(argv[i]);
            run_hybrid_query<float>(data,filters_file,queries_file,
                query_filters_file,gt_file,index_folder,P,
                shared,single);
        }
    }
    else {
        cout << "Unknown vector_type" << endl;
    }
    
    return 0;
}
