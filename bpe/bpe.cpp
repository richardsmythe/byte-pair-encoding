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
		}
		else {
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
		pairs.push_back(Pair{ i, 0 });
	}

	// tokenise input text
	for (char c : text) {
		tokens_in.push_back(static_cast<uint8_t>(c));
	}

	// BPE merge loop
	while (true) {
		freq.clear();
		for (size_t i = 0; i + 1 < tokens_in.size(); i++) {
			Pair pair{ tokens_in[i], tokens_in[i + 1] };
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
				Pair pair{ tokens_in[i], tokens_in[i + 1] };
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
		std::cerr << "\nError writing lookup table" << std::endl;
		return;
	}
	for (size_t i = 0; i < pairs.size(); i++) {
		Pair p = pairs[i];
		if (p.r == 0) {
			if (p.l == '[') {
				out << i << ": 0x" << std::hex << std::uppercase << (int)p.l << std::dec << "\n";
			}
			else if (p.l >= 32 && p.l <= 126) {
				out << i << ": '" << static_cast<char>(p.l) << "'\n";
			}
			else {
				out << i << ": 0x" << std::hex << std::uppercase << (int)p.l << std::dec << "\n";
			}
		}
		else {
			out << i << ": [" << p.l << ", " << p.r << "]\n";
		}
	}
	out.close();
}

PairArray decompress_using_lookup_table(const std::string& filename) {
	std::ifstream file(filename);
	PairArray pairs;
	std::string line;
	while (std::getline(file, line)) {
		size_t colon = line.find(':');
		if (colon == std::string::npos) continue;
		uint32_t id = std::stoi(line.substr(0, colon));
		Pair p;
		if (line.find('[') != std::string::npos) {
			// format: ID:[l,r]
			size_t l_start = line.find('[') + 1;
			size_t comma = line.find(',', l_start);
			size_t r_end = line.find(']', comma);

			if (comma == std::string::npos || r_end == std::string::npos || l_start >= comma || comma + 1 >= r_end) {
				std::cerr << "Error: Malformed line in lookup table: " << line << std::endl;
				continue; 
			}

			std::string l_str = line.substr(l_start, comma - l_start);
			std::string r_str = line.substr(comma + 1, r_end - comma - 1);

			try {
				uint32_t l = std::stoul(l_str);  // left value of token pair
				uint32_t r = std::stoul(r_str); // right value of token pair
				p = { l,r };
			}
			catch (const std::invalid_argument& e) {
				std::cerr << "Error: Invalid argument in stoul: " << e.what() << " Line: " << line << std::endl;
				continue; 
			}
			catch (const std::out_of_range& e) {
				std::cerr << "Error: Out of range in stoul: " << e.what() << " Line: " << line << std::endl;
				continue; 
			}
		}
		else if (line.find('\'') != std::string::npos) {
			// parse single character
			size_t quote = line.find('\'') + 1;
			char c = line[quote];
			p = { static_cast<uint32_t>(c),0 };
		}
		else if (line.find("0x") != std::string::npos) {
			//format: ID: 0xXX
			size_t hex_start = line.find("0x") + 2;
			size_t hex_end = line.find_first_not_of("0123456789ABCDEF", hex_start);
			std::string hex_str = line.substr(hex_start, hex_end - hex_start);
			uint32_t val = std::stoul(hex_str, nullptr, 16);
			p = { val, 0 };
		}
		else {
			continue;
		}
		if (pairs.size() <= id) pairs.resize(id + 1);
		pairs[id] = p;
	}
	return pairs;
}

std::string expand_token(const PairArray& pairs, uint32_t token) {
	if (pairs[token].r == 0) {
		return std::string(1, static_cast<char>(pairs[token].l));
	}
	else {
		return expand_token(pairs, pairs[token].l) + expand_token(pairs, pairs[token].r);
	}
}

std::string decode_tokens(const PairArray& pairs, const Uint32Array& tokens) {
	std::string output;
	for (size_t i = 0; i < tokens.size(); i++)
	{
		output += expand_token(pairs, tokens[i]);
	}
	return output;
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
	std::string lookup_table_filename = "lookup_table.txt";


	run_bpe(text, pairs, tokens_out);
	std::cout << "\nReadable compressed view:\n";
	dump_tokens(pairs, tokens_out);
	std::cout << "\nCompressed token IDs:\n";
	print_compressed_tokens(tokens_out);
	write_lookup_table(lookup_table_filename, pairs);
	std::cout << "\nLookup table written to 'lookup_table.txt'\n";
	
	// decode and print original input
	pairs.clear();
	std::cout << "\nDecoding tokens... Reading lookup table and converting to original output";
	pairs = decompress_using_lookup_table(lookup_table_filename);
	std::string original_input_text = decode_tokens(pairs, tokens_out);
	std::cout << "\n\n\n" << original_input_text << std::endl;


	return 0;
}

