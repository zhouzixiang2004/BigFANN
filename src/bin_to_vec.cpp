#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"

auto convert_onebyte(const char* infile, const char* outfile) {
    auto str = parlay::chars_from_file(infile);
    int n = *((int *) str.data());
    int dims = *(((int *) str.data()) + 1);
    std::cout << "n = " << n << " d = " << dims << std::endl;
    parlay::sequence<char> head(4);
    *((int *) head.data()) = dims;
    auto vects = parlay::tabulate(n, [&] (size_t i) {
		return parlay::append(head,parlay::to_sequence(str.cut(8 + i * dims, 8 + (i+1) * dims)));});
    parlay::chars_to_file(parlay::flatten(vects), outfile);
}

auto convert_fourbyte(const char* infile, const char* outfile) {
    auto str = parlay::chars_from_file(infile);
    int n = *((int *) str.data());
    int dims = *(((int *) str.data()) + 1);
    std::cout << "n = " << n << " d = " << dims << std::endl;
    parlay::sequence<char> head(4);
    *((int *) head.data()) = dims;
    auto vects = parlay::tabulate(n, [&] (size_t i) {
		return parlay::append(head,parlay::to_sequence(str.cut(8 + i * (4*dims), 8 + (i+1) * (4*dims))));});
    parlay::chars_to_file(parlay::flatten(vects), outfile);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cout << "usage: bin_to_vec type <infile> <outfile>" << std::endl;
    return 1;
  }
  std::string tp = std::string(argv[1]);
  if(tp == "uint8") convert_onebyte(argv[2], argv[3]);
  else if(tp == "float" || tp == "int") convert_fourbyte(argv[2], argv[3]);
  else{
    std::cout << "invalid type: specify uint8, float, or int" << std::endl;
    abort();
  }
  return 0;
}