#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

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
			.items = malloc(item_size * initial_capacity)
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
	uint32_t pair[2];
} Pair;

uint32_t next_token_id = 256;
uint32_t* merge_map = NULL;

typedef struct {
	Pair key;
	size_t value;
} Freq;

Freq* freq = NULL;

int compare_freqs(const void* a, const void* b) {
	const Freq* af = a;
	const Freq* bf = b;
	return (int)(bf->value - af->value);
}

int main() {
	const char* text = "The original BPE algorithm operates by iteratively replacing the most common contiguous sequences of characters in a target text with unused 'placeholder' bytes. The iteration ends when no sequences can be found, leaving the target text effectively compressed. Decompression can be performed by reversing this process, querying known placeholder terms against their corresponding denoted sequence, using a lookup table. In the original paper, this lookup table is encoded and stored alongside the compressed text.";
	int text_size = strlen(text);

	Array tokens = array_init(sizeof(uint32_t), text_size);
	for (size_t i = 0; i < strlen(text); i++) {
		uint32_t token = (uint8_t)text[i];
		array_append(&tokens, &token);
	}


	uint32_t* token_items = (uint32_t*)tokens.items;

	for (size_t i = 0; i < tokens.length - 1; i++) {
		Pair pair = { .pair = { token_items[i], token_items[i + 1] } };

		ptrdiff_t index = hmgeti(freq, pair);
		if (index < 0) {
			hmput(freq, pair, 1);
		}
		else {
			freq[index].value += 1;
		}
	}

	Array sorted_freqs = array_init(sizeof(Freq), 16);
	for (ptrdiff_t i = 0; i < hmlen(freq); i++) {
		array_append(&sorted_freqs, &freq[i]);
	}

	qsort(sorted_freqs.items, sorted_freqs.length, sizeof(Freq), compare_freqs);

	for (size_t i = 0; i < 10 && i < sorted_freqs.length; i++) {
		Freq* f = &((Freq*)sorted_freqs.items)[i];
		printf("[%u %u] => %zu\n", (char)f->key.pair[0], (char)f->key.pair[1], f->value);
	}

	free(sorted_freqs.items);
	free(tokens.items);
	hmfree(freq);

	return 0;
}
