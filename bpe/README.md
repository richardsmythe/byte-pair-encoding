# BPE Tokenizer & SMS Spam Classifier (C++)

## Overview
This project demonstrates a custom implementation of Byte Pair Encoding (BPE) in C++. BPE is a powerful text tokenization technique used in modern NLP to break text into subword units, enabling efficient handling of rare words and reducing vocabulary size. To showcase the effectiveness of BPE, the project includes a simple SMS spam classifier using a neural network.

## What is BPE?
- **Byte Pair Encoding (BPE)** is a data compression and tokenization algorithm.
- It iteratively replaces the most frequent pair of bytes (or characters) in a text with a new token, building a vocabulary of subword units.
- BPE is widely used in NLP models (like GPT and BERT) to handle out-of-vocabulary words and improve text representation.

## Project Features
- **Custom BPE implementation in C++** for tokenizing SMS messages.
- **Builds a BPE vocabulary and lookup table** from the dataset, and saves as text file in the project directory.
- **Tokenizes each message into BPE subword tokens.**
- **Demonstrates BPE output** by converting messages into sequences of token IDs.
- **Shows how BPE tokens can be used as features** for downstream machine learning tasks.
- **Includes a simple neural network classifier** to illustrate how BPE tokenization can be used for spam detection.

## How it Works
1. **BPE Tokenization:**
   - Reads SMS messages and labels from a file.
   - Builds a BPE vocabulary and lookup table.
   - Tokenizes each message into BPE subword tokens and converts them to token IDs.
2. **Feature Engineering (Demo):**
   - Maps each token ID to a random embedding vector.
   - Averages all token embeddings in a message to get a single feature vector.
3. **Model Training (Demo):**
   - Trains a simple neural network to classify messages as spam or ham using BPE-based features.
4. **Evaluation:**
   - Prints accuracy, precision, recall, F1 score, and confusion matrix on the test set.

## Why Use BPE?
- Handles rare and out-of-vocabulary words by breaking them into known subwords.
- Reduces vocabulary size, making models more efficient.
- Improves generalization in NLP tasks.

## Example BPE Lookup Table
Below is a sample of what a BPE lookup table might look like after training:
0: 0x0
1: 0x1
2: 0x2
...
32: ' '
33: '!'
...
97: 'a'
98: 'b'
...
256: [116, 115]
257: [256, 32]- Entries 0–255: Base tokens, each representing a single byte/character.
- Entries 256 and above: New tokens created by merging frequent pairs during BPE training. For example, `256: [116, 115]` means token 256 represents the pair 't' and 's'.
- The lookup table allows you to decode any token ID back to its original character sequence by recursively expanding the pairs.

## Classifier Output

Test accuracy: 88.6894%

Confusion Matrix:
TP: 137  FP: 106
FN: 20  TN: 851
Precision: 0.563786
Recall: 0.872611
F1 Score: 0.685

## Usage
1. Place your SMS dataset in the project directory (e.g., `SMSSpamCollection.txt`).
2. Build and run the project.
3. View BPE tokenization output and classifier results in the console.

---

*This project is for educational purposes and demonstrates BPE tokenization and its application in NLP using C++.*
