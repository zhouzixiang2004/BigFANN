CC := gcc
CXX := g++
BIN := ../bin
INCLUDES := -I./
LIBS := -lgomp

CFLAGS    := -Wall -fopenmp
CXXFLAGS  := -Wall -fopenmp -std=c++20 -march=native
CXXFLAGS += -Wno-narrowing
ICPCFLAGS := -Wall -qopenmp

ifeq ($(PAPI), 1)
  CFLAGS += -DUSE_PAPI
  CXXFLAGS += -DUSE_PAPI
  INCLUDES += -I$(PAPI_HOME)/include
  LIBS += -L$(PAPI_HOME)/lib -lpapi
endif

ifeq ($(DEBUG), 1)
  CFLAGS += -g -O0 
  CXXFLAGS += -g -O0 
  NVFLAGS += -G
else
  CFLAGS += -O3 -g
  CXXFLAGS += -O3 -g
  NVFLAGS += -O3 -w
endif

ifeq ($(PREFETCH), 1)
  CXXFLAGS += -DUSE_PF_VEC
  CXXFLAGS += -DUSE_PF_GRAPH
endif

ifeq ($(PARLAY), 1)
  CXXFLAGS += -DUSE_PARLAY
endif

VPATH += ./cpp/
OBJS = VertexSet.o graph.o
DEPS = bfs.hpp ivf.hpp utils.hpp graph.hpp

CFLAGS += ffast-math -march=native

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

all: preprocess build_index run_query

preprocess: vec_to_bin bin_to_vec gen_synthetic_filters get_groundtruth

vec_to_bin: vec_to_bin.o
	g++ -pthread vec_to_bin.cpp -o $(BIN)/vec_to_bin -std=c++17

bin_to_vec: bin_to_vec.o
	g++ -pthread bin_to_vec.cpp -o $(BIN)/bin_to_vec -std=c++17

gen_synthetic_filters: gen_synthetic_filters.o
	$(CXX) $(CXXFLAGS) gen_synthetic_filters.o -o $(BIN)/gen_synthetic_filters

get_groundtruth: VertexSet.o get_groundtruth.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) VertexSet.o get_groundtruth.o -o $(BIN)/get_groundtruth

build_index: $(OBJS) build_index.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) build_index.o -o $(BIN)/build_index

run_query: $(OBJS) run_query.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) run_query.o -o $(BIN)/run_query


clean:
	rm *.o
