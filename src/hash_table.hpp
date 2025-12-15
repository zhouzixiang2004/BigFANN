#pragma once

#include <vector>
#include <stdint.h>
#include <string.h>
#include "common.hpp"

template <typename T>
class hash_filter {
private:
  int bits;
  int capacity;
  std::vector<T> seen;
  std::hash<T> hasher;
/*
parlay uses:
inline uint64_t hash64_2(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}
*/
public:
  hash_filter(int L) {
    bits = __lg(L)+1; // round up to power of 2
    L = (1 << bits);
    capacity = L;
    seen.resize(L);
    fill(seen.begin(),seen.end(),-1); // note: -1 should not be a possible index
  }
  int get_hash(T thing) {
    //return hasher(thing) & ((1 << bits)-1);
    return thing & ((1 << bits)-1);
  }
  bool contains(T thing) {
    int h = get_hash(thing);
    return (seen[h] == thing);
  }
  bool add(T thing) { // returns true if successfully added
    int h = get_hash(thing);
    if (seen[h] == thing) return false;
    seen[h] = thing;
    return true;
  }
};

template <typename T>
class hash_subgroup {
private:
    int bits;
    std::vector<T> group;
    std::hash<T> hasher;
    std::vector<int> seen;  // Stores indices or -1 if not present

    // Get hash value (masking to fit within table size)
    int get_hash(T thing) {
        return hasher(thing) & ((1 << bits) - 1);
    }

public:
    hash_subgroup(const std::vector<T> &_group) {
        group = _group;
        bits = __lg(group.size()) + 2;  // Determine number of bits based on size
        seen.resize(1 << bits, -1);  // Initialize table with -1 (indicating empty)

        // Build the open addressing map
        int worst = 0;
        for (size_t i = 0; i < group.size(); ++i) {
            size_t hash_value = get_hash(group[i]);
            size_t probe_idx = hash_value;

            // Handle collisions using linear probing
            int num = 0;
            while (seen[probe_idx] != -1) {
                // Linear probing (wrap around if necessary)
                probe_idx = (probe_idx + 1) & ((1 << bits) - 1);  // Ensures we stay within bounds
                num++;
            }
            if (num > worst) worst = num;

            // Insert the item at the found position
            seen[probe_idx] = i;
        }
        //cout << "Built hash table, worst case " << worst << "/" << group.size() << endl;
    }

    T operator[](size_t i) {
        return group[i];
    }

    size_t pos(const T &i) {
        size_t hash_value = get_hash(i);
        size_t probe_idx = hash_value;

        // Linear probing to find the correct position of i
        while (seen[probe_idx] != -1) {
            if (group[seen[probe_idx]] == i) {
                return seen[probe_idx];  // Return the found index
            }
            // If not the right item, keep probing
            probe_idx = (probe_idx + 1) & ((1 << bits) - 1);
        }

        // If item not found, return a special value (e.g., -1)
        return -1;
    }
};