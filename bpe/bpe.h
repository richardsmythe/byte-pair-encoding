#ifndef BPE_H
#define BPE_H

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>

namespace bpe {

struct Pair {
    uint32_t l, r;
    bool operator==(const Pair& other) const;
};

using PairArray = std::vector<Pair>;
using Uint32Array = std::vector<uint32_t>;

void dump_tokens(const PairArray& pairs, const Uint32Array& tokens);
void swap_tokens(Uint32Array& a, Uint32Array& b);
void run_bpe(const std::string& text, PairArray& pairs, Uint32Array& tokens_out);
void print_compressed_tokens(const Uint32Array& tokens);
void write_lookup_table(const std::string& filename, const PairArray& pairs);
PairArray decompress_using_lookup_table(const std::string& filename);
std::string expand_token(const PairArray& pairs, uint32_t token);
std::string decode_tokens(const PairArray& pairs, const Uint32Array& tokens);

} 

namespace std {
    template<>
    struct hash<bpe::Pair> {
        size_t operator()(const bpe::Pair& p) const;
    };
}

#endif 
