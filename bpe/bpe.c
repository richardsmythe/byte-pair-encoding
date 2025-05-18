#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

typedef struct {
	uint32_t l, r;
} Pair;

typedef struct {
	Pair key;
	size_t value;
} Freq;

typedef struct {
	size_t length;
	size_t capacity;
	size_t item_size;
	void* items;
} Array;

Array array_init(size_t item_size, size_t initial_capacity) {
	return (Array) {
		.length = 0,
			.capacity = initial_capacity,
			.item_size = item_size,
			.items = malloc(item_size * initial_capacity),
	};
}

void array_append(Array* a, void* item) {
	if (a->length >= a->capacity) {
		a->capacity *= 2;
		void* new_items = realloc(a->items, a->item_size * a->capacity);
		if (!new_items) {
			fprintf(stderr, "error appending to array\n");
			exit(1);
		}
		a->items = new_items;
	}
	void* target = (char*)a->items + (a->length * a->item_size);
	memcpy(target, item, a->item_size);
	a->length += 1;
}

typedef struct {
	Array a;
} PairArray;

typedef struct {
	Array a;
} Uint32Array;

#define pair_array_init()     ((PairArray){ array_init(sizeof(Pair), 256) })
#define uint32_array_init()   ((Uint32Array){ array_init(sizeof(uint32_t), 256) })

#define pair_array_append(pa, val)      array_append(&(pa)->a, (val))
#define uint32_array_append(ua, val)    array_append(&(ua)->a, (val))

#define pair_array_get(pa, i)   (&((Pair*)(pa).a.items)[i])
#define uint32_array_get(ua, i) (&((uint32_t*)(ua).a.items)[i])

void dump_tokens(PairArray pairs, Uint32Array tokens) {
	uint32_t* token_items = (uint32_t*)tokens.a.items;
	Pair* pair_items = (Pair*)pairs.a.items;

	for (size_t i = 0; i < tokens.a.length; i++) {
		uint32_t token = token_items[i];
		assert(token < pairs.a.length);
		Pair p = pair_items[token];

		if (p.r == 0) {
			printf("%c", (char)p.l);
		}
		else {
			printf("[%u]", token);
		}
	}
	printf("\n");
}

void swap_tokens(Uint32Array* a, Uint32Array* b) {
	Uint32Array temp = *a;
	*a = *b;
	*b = temp;
	b->a.length = 0;
}


void run_bpe(const char* text, PairArray* pairs, Uint32Array* tokens_out) {
	int text_size = strlen(text);
	Freq* freq = NULL;

	Uint32Array tokens_in = uint32_array_init();
	Uint32Array temp_tokens = uint32_array_init();

	// add base tokens for all 0-255 values
	for (uint32_t i = 0; i < 256; ++i) {
		Pair p = { .l = i, .r = 0 };
		pair_array_append(pairs, &p);
	}

	// tokenise input text
	for (int i = 0; i < text_size; ++i) {
		uint32_t t = (uint8_t)text[i];
		uint32_array_append(&tokens_in, &t);
	}

	// BPE merge loop
	while (1) {
		hmfree(freq);
		freq = NULL;

		uint32_t* token_items = (uint32_t*)tokens_in.a.items;
		for (size_t i = 0; i < tokens_in.a.length - 1; i++) {
			Pair pair = { .l = token_items[i], .r = token_items[i + 1] };
			ptrdiff_t index = hmgeti(freq, pair);
			if (index < 0) {
				hmput(freq, pair, 1);
			}
			else {
				freq[index].value += 1;
			}
		}

		if (hmlen(freq) == 0) break;

		ptrdiff_t max_index = 0;
		for (ptrdiff_t i = 1; i < hmlen(freq); i++) {
			if (freq[i].value > freq[max_index].value) {
				max_index = i;
			}
		}

		if (freq[max_index].value <= 1) break;

		pair_array_append(pairs, &freq[max_index].key);

		temp_tokens.a.length = 0;
		for (size_t i = 0; i < tokens_in.a.length;) {
			if (i + 1 < tokens_in.a.length) {
				Pair pair = { .l = token_items[i], .r = token_items[i + 1] };
				if (memcmp(&pair, &freq[max_index].key, sizeof(pair)) == 0) {
					uint32_t new_token = (uint32_t)(pairs->a.length - 1);
					uint32_array_append(&temp_tokens, &new_token);
					i += 2;
					continue;
				}
			}
			uint32_array_append(&temp_tokens, &token_items[i]);
			i += 1;
		}
		swap_tokens(&tokens_in, &temp_tokens);
	}

	*tokens_out = tokens_in;
	free(temp_tokens.a.items);
	hmfree(freq);
}


void print_compressed_tokens(Uint32Array* tokens) {
	uint32_t* token_items = (uint32_t*)tokens->a.items;
	for (size_t i = 0; i < tokens->a.length; i++) {
		printf("%u ", token_items[i]);
	}
	printf("\n");
}

void write_lookup_table(const char* filename, PairArray* pairs) {
	FILE* out = fopen(filename, "w");
	if (!out) {
		perror("Error writing lookup table");
		return;
	}

	Pair* pair_items = (Pair*)pairs->a.items;

	for (size_t i = 0; i < pairs->a.length; i++) {
		Pair p = pair_items[i];
		if (p.r == 0) {
			// original character
			if (p.l >= 32 && p.l <= 126) {
				fprintf(out, "%zu: '%c'\n", i, (char)p.l);
			}
			else {
				fprintf(out, "%zu: 0x%02X\n", i, p.l);  // hex
			}
		}
		else {
	
			fprintf(out, "%zu: [%u, %u]\n", i, p.l, p.r);
		}
	}

	fclose(out);
}


int main() {
	char filename[512];

	printf("Enter the input file path: ");
	if (!fgets(filename, sizeof(filename), stdin)) {
		fprintf(stderr, "Error reading input.\n");
		return 1;
	}

	filename[strcspn(filename, "\n")] = 0;
	FILE* file = fopen(filename, "rb");
	if (!file) {
		perror("Error opening file");
		return 1;
	}

	fseek(file, 0, SEEK_END);
	long file_size = ftell(file);
	rewind(file);

	char* text = (char*)malloc(file_size + 1);
	if (!text) {
		fprintf(stderr, "Memory allocation failed\n");
		fclose(file);
		return 1;
	}

	size_t bytes_read = fread(text, 1, file_size, file);
	fclose(file);
	text[bytes_read] = '\0';

	// run bpe
	PairArray pairs = pair_array_init();
	Uint32Array tokens_out;
	run_bpe(text, &pairs, &tokens_out);

	printf("\n\tReadable compressed view:\n");
	dump_tokens(pairs, tokens_out);

	printf("\n\tCompressed token IDs:\n");
	print_compressed_tokens(&tokens_out);

	// create the lookup table for decompression
	write_lookup_table("lookup_table.txt", &pairs);
	printf("\n\tLookup table written to 'lookup_table.txt'\n");


	// Cleanup
	free(text);
	free(pairs.a.items);
	free(tokens_out.a.items);

	return 0;
}

