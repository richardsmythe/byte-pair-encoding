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

// Generic array backend
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

int main() {
	const char* text = "The original BPE algorithm operates by iteratively replacing the most common contiguous sequences of characters in a target text with unused 'placeholder' bytes. The iteration ends when no sequences can be found, leaving the target text effectively compressed. Decompression can be performed by reversing this process, querying known placeholder terms against their corresponding denoted sequence, using a lookup table. In the original paper, this lookup table is encoded and stored alongside the compressed text.";
	int text_size = strlen(text);
	Freq* freq = NULL; // hash table for frequencies of characters
	PairArray pairs = pair_array_init(); // array for pairs of characters
	Uint32Array tokens_in = uint32_array_init(); // array to store tokens - initial characters
	Uint32Array tokens_out = uint32_array_init(); // reused for each merge output

	// add all possible 0-255 characters as base tokens
	for (uint32_t i = 0; i < 256; ++i) {
		Pair p = { .l = i };
		pair_array_append(&pairs, &p);
	}

	// tokenize the input text as individual character tokens
	for (int i = 0; i < text_size; ++i) {
		uint32_t t = (uint8_t)text[i];
		uint32_array_append(&tokens_in, &t);
	}

	// start BPE merge
	while (1) {

		// clear previous frequencies
		hmfree(freq);
		freq = NULL;

		// calculate frequencies of adjacent token pairs
		uint32_t* token_items = (uint32_t*)tokens_in.a.items;
		for (size_t i = 0; i < tokens_in.a.length - 1; i++) {
			Pair pair = {
				.l = token_items[i],
				.r = token_items[i + 1],
			};

			ptrdiff_t index = hmgeti(freq, pair);
			if (index < 0) {
				hmput(freq, pair, 1);
			}
			else {
				freq[index].value += 1;
			}
		}

		// stop if no frequent pairs are left
		if (hmlen(freq) == 0) break;

		// get most frequent pair
		ptrdiff_t max_index = 0;
		for (ptrdiff_t i = 1; i < hmlen(freq); i++) {
			if (freq[i].value > freq[max_index].value) {
				max_index = i;
			}
		}

		// stop if frequency is 1, cannot compress anymore
		if (freq[max_index].value <= 1) break;

		//printf("[%u %u] => %zu\n",
		//	freq[max_index].key.l,
		//	freq[max_index].key.r,
		//	freq[max_index].value);

		pair_array_append(&pairs, &freq[max_index].key);

		// merge the most frequent pair into a new token
		tokens_out.a.length = 0;
		for (size_t i = 0; i < tokens_in.a.length;) {
			if (i + 1 < tokens_in.a.length) {
				Pair pair = {
					.l = token_items[i],
					.r = token_items[i + 1]
				};

				if (memcmp(&pair, &freq[max_index].key, sizeof(pair)) == 0) {
					uint32_t new_token = (uint32_t)(pairs.a.length - 1);
					uint32_array_append(&tokens_out, &new_token);
					i += 2;
					continue;
				}
			}
			uint32_array_append(&tokens_out, &token_items[i]);
			i += 1;
		}
		swap_tokens(&tokens_in, &tokens_out);	
	}


	dump_tokens(pairs, tokens_in);

	free(pairs.a.items);
	free(tokens_in.a.items);
	free(tokens_out.a.items);  
	hmfree(freq);

	return 0;
}
