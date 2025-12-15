// Borrowed from ParlayANN
#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"

// convert from .bvec file to .u8bin file


auto convert_onebyte(const char* infile, const char* outfile, int bound = INT_MAX) {
  auto start = parlay::chars_from_file(infile,false,0,4);
  int dims = *((int *) start.data());
  std::cout << "d = " << dims << std::endl;
  auto str = parlay::chars_from_file(infile,false,0,(std::streamoff) bound*(4+dims));
  int n = str.size()/(dims+4);
  n = std::min(n,bound);
  std::cout << "n = " << n << " d = " << dims << std::endl;
  auto vects = parlay::tabulate(n, [&] (size_t i) {
		     return parlay::to_sequence(str.cut(4 + i * (4 + dims), (i+1) * (4 + dims)));});
  parlay::sequence<char> head(8);
  *((int *) head.data()) = n;
  *(((int *) head.data()) + 1) = dims;
  auto strout = parlay::append(head, parlay::flatten(vects));
  parlay::chars_to_file(strout, outfile);
}

auto convert_fourbyte(const char* infile, const char* outfile, int bound = INT_MAX) {
  auto start = parlay::chars_from_file(infile,false,0,4);
  int dims = *((int *) start.data());
  std::cout << "d = " << dims << std::endl;
  auto str = parlay::chars_from_file(infile,false,0,(std::streamoff) bound*(4+4*dims));
  int n = str.size()/(4*dims+4);
  n = std::min(n,bound);
  std::cout << "n = " << n << " d = " << dims << std::endl;
  auto vects = parlay::tabulate(n, [&] (size_t i) {
		     return parlay::to_sequence(str.cut(4 + i * (4 + 4*dims), (i+1) * (4 + 4*dims)));});
  parlay::sequence<char> head(8);
  *((int *) head.data()) = n;
  *(((int *) head.data()) + 1) = dims;
  auto strout = parlay::append(head, parlay::flatten(vects));
  parlay::chars_to_file(strout, outfile);
}

int main(int argc, char* argv[]) {
  if (argc != 4 && argc != 5) {
    std::cout << "usage: vec_to_bin type <infile> <outfile> <bound(optional)>" << std::endl;
    return 1;
  }
  std::string tp = std::string(argv[1]);
  if(tp == "uint8") {
    if (argc == 5) convert_onebyte(argv[2], argv[3], atoi(argv[4]));
    else convert_onebyte(argv[2], argv[3]);
  }
  else if(tp == "float" || tp == "int") {
    if (argc == 5) convert_fourbyte(argv[2], argv[3], atoi(argv[4]));
    else convert_fourbyte(argv[2], argv[3]);
  }
  else{
    std::cout << "invalid type: specify uint8, float, or int" << std::endl;
    abort();
  }
  return 0;
}
