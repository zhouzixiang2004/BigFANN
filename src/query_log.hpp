#pragma once
#include <string>

struct query_info {
    int id;
    int distance_comps;
    double time;
    double distance_time;
    int recall;
    int filter_count_a;
    int filter_count_b;
    std::string cas;
    // graph
    int iter;
    int hops;
    int push;
    // ivf
    int checks;
    int pass_filter;
    int clusters;
    double filter_time;
};

std::vector<query_info> query_log;