#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <cassert>

struct Pair {
    uint32_t l, r;
    bool operator==(const Pair& other) const {
        return l == other.l && r == other.r;
    }
};

namespace std {
    template<>
    struct hash<Pair> {
        size_t operator()(const Pair& p) const {
            return hash<uint32_t>()(p.l) ^ (hash<uint32_t>()(p.r) << 1);
        }
    };
}

using PairArray = std::vector<Pair>;
using Uint32Array = std::vector<uint32_t>;

void dump_tokens(const PairArray& pairs, const Uint32Array& tokens) {
    for (size_t i = 0; i < tokens.size(); i++) {
        uint32_t token = tokens[i];
        assert(token < pairs.size());
        Pair p = pairs[token];
        if (p.r == 0) {
            std::cout << static_cast<char>(p.l);
        } else {
            std::cout << "[" << token << "]";
        }
    }
    std::cout << std::endl;
}

void swap_tokens(Uint32Array& a, Uint32Array& b) {
    std::swap(a, b);
    b.clear();
}

void run_bpe(const std::string& text, PairArray& pairs, Uint32Array& tokens_out) {
    std::unordered_map<Pair, size_t> freq;
    Uint32Array tokens_in;
    Uint32Array temp_tokens;

    // add base tokens for all 0-255 values
    for (uint32_t i = 0; i < 256; ++i) {
        pairs.push_back(Pair{i, 0});
    }

    // tokenise input text
    for (char c : text) {
        tokens_in.push_back(static_cast<uint8_t>(c));
    }

    // BPE merge loop
    while (true) {
        freq.clear();
        for (size_t i = 0; i + 1 < tokens_in.size(); i++) {
            Pair pair{tokens_in[i], tokens_in[i + 1]};
            freq[pair]++;
        }
        if (freq.empty()) break;
        auto max_it = freq.begin();
        for (auto it = freq.begin(); it != freq.end(); ++it) {
            if (it->second > max_it->second) {
                max_it = it;
            }
        }
        if (max_it->second <= 1) break;
        std::cout << "Tokens before merge: " << tokens_in.size() << std::endl;
        pairs.push_back(max_it->first);
        std::cout << "Merged most frequent pair: [" << max_it->first.l << "," << max_it->first.r << "] => token ID: " << pairs.size() - 1 << std::endl;
        temp_tokens.clear();
        for (size_t i = 0; i < tokens_in.size();) {
            if (i + 1 < tokens_in.size()) {
                Pair pair{tokens_in[i], tokens_in[i + 1]};
                if (pair == max_it->first) {
                    temp_tokens.push_back(static_cast<uint32_t>(pairs.size() - 1));
                    i += 2;
                    continue;
                }
            }
            temp_tokens.push_back(tokens_in[i]);
            i += 1;
        }
        swap_tokens(tokens_in, temp_tokens);
    }
    tokens_out = tokens_in;
}

void print_compressed_tokens(const Uint32Array& tokens) {
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i] << " ";
    }
    std::cout << std::endl;
}

void write_lookup_table(const std::string& filename, const PairArray& pairs) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error writing lookup table" << std::endl;
        return;
    }
    for (size_t i = 0; i < pairs.size(); i++) {
        Pair p = pairs[i];
        if (p.r == 0) {
            if (p.l >= 32 && p.l <= 126) {
                out << i << ": '" << static_cast<char>(p.l) << "'\n";
            } else {
                out << i << ": 0x" << std::hex << std::uppercase << (int)p.l << std::dec << "\n";
            }
        } else {
            out << i << ": [" << p.l << ", " << p.r << "]\n";
        }
    }
    out.close();
}

int main() {
    std::string filename;
    std::cout << "Enter the input file path: ";
    std::getline(std::cin, filename);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    PairArray pairs;
    Uint32Array tokens_out;
    run_bpe(text, pairs, tokens_out);
    std::cout << "\n\tReadable compressed view:\n";
    dump_tokens(pairs, tokens_out);
    std::cout << "\n\tCompressed token IDs:\n";
    print_compressed_tokens(tokens_out);
    write_lookup_table("lookup_table.txt", pairs);
    std::cout << "\n\tLookup table written to 'lookup_table.txt'\n";
    return 0;
}

