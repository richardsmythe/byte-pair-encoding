#include "data_handler.h"
#include "nn.h"
#include <iostream>
#include "bpe.h"
#include <sstream>
#include <algorithm>

int main() {
    const size_t INPUT_SIZE = 32;
    const size_t ITERATIONS = 4000;
    size_t TOP_N = 200;

    // load dataset and preprocess
    Data_Handler dh;
    dh.read_csv("SMSSpamCollection.txt", "\t");
    dh.split_data(); 

    // Get vocab size and estimate best TOP_N
	size_t vocab_size = dh.get_vocabulary_size();
    if (vocab_size <= 100) {
        TOP_N = vocab_size;
    }
    else {
        const double PERCENT = 0.05;    // select top 5% by default
        const size_t MIN_TOP = 50;      // at least 50 features for medium vocabs
        const size_t MAX_TOP = 2000;    // cap to avoid too many features
        size_t candidate = static_cast<size_t>(vocab_size * PERCENT);
        if (candidate < MIN_TOP) candidate = MIN_TOP;
        if (candidate > MAX_TOP) candidate = MAX_TOP;
        TOP_N = std::min(candidate, vocab_size);
    }

    // Select top N features using chi-square
    std::vector<uint32_t> selected_features = dh.select_features_chi_square(TOP_N);
    std::vector<std::vector<float>> train_features, test_features;
    std::vector<float> train_labels, test_labels;

    // prepare training data with embeddings, filtering by selected features
    for (const auto& d : dh.get_training_data()) {
        std::vector<uint32_t> filtered;
        for (auto t : d.get_feature_vector()) {
            if (std::find(selected_features.begin(), selected_features.end(), t) != selected_features.end()) {
                filtered.push_back(t);
            }
        }
        train_features.push_back(dh.embed_and_average(filtered, INPUT_SIZE));
        train_labels.push_back(static_cast<float>(d.get_label()));
    }

    // prepare test data with embeddings, filtering by selected features
    for (const auto& d : dh.get_test_data()) {
        std::vector<uint32_t> filtered;
        for (auto t : d.get_feature_vector()) {
            if (std::find(selected_features.begin(), selected_features.end(), t) != selected_features.end()) {
                filtered.push_back(t);
            }
        }
        test_features.push_back(dh.embed_and_average(filtered, INPUT_SIZE));
        test_labels.push_back(static_cast<float>(d.get_label()));
    }

    // Demonstrate cosine similarity between messages
    std::cout << "\n--- Cosine Similarity Demonstration ---\n";
    
    // Find spam and ham examples
    int spam_index = -1, ham_index = -1, another_spam_index = -1;
    for (size_t i = 0; i < test_labels.size(); ++i) {
        if (test_labels[i] == 1.0f && spam_index == -1) {
            spam_index = i;
        } else if (test_labels[i] == 0.0f && ham_index == -1) {
            ham_index = i;
        } else if (test_labels[i] == 1.0f && another_spam_index == -1 && spam_index != -1) {
            another_spam_index = i;
            break;
        }
        
        if (spam_index != -1 && ham_index != -1 && another_spam_index != -1) {
            break;
        }
    }
    
    if (spam_index != -1 && ham_index != -1 && another_spam_index != -1) {
        // Cosine similarity between spam and ham
        float sim_spam_ham = dh.cosine_similarity(test_features[spam_index], test_features[ham_index]);
        // Cosine similarity between two spam messages
        float sim_spam_spam = dh.cosine_similarity(test_features[spam_index], test_features[another_spam_index]);
        
        std::cout << "Cosine similarity between spam and ham: " << sim_spam_ham << std::endl;
        std::cout << "Cosine similarity between two spam messages: " << sim_spam_spam << std::endl;
        std::cout << "Note: Higher values indicate more similar messages\n";
    }

    // calculate class weights for weighted loss
    // weights for each class is inversely proportional to its frequency
    float weight_ham = 1.0f, weight_spam = 1.0f; // higher for spam as it's minority class
    if (dh.is_training_imbalanced()) {        
        std::cout << "Dataset is imbalanced. Rebalancing...\n";
        weight_ham = (float)(dh.ham_count + dh.spam_count) / (2.0f * dh.ham_count);
        weight_spam = (float)(dh.ham_count + dh.spam_count) / (2.0f * dh.spam_count);
        std::cout << "Using weighted loss: weight_ham=" << weight_ham << ", weight_spam=" << weight_spam << std::endl;
    }

    // train the NN
    NeuralNetwork nn(INPUT_SIZE, ITERATIONS);
    nn.train(train_features, train_labels, 0.1f, weight_ham, weight_spam);

    // analyse results
    size_t correct = 0;
    for (size_t i = 0; i < test_features.size(); ++i) {
        float prediction = nn.predict(test_features[i]);
        int pred_label;
        if (prediction > 0.5f) {
            pred_label = 1;
        } else {
            pred_label = 0;
        }
        if (pred_label == static_cast<int>(test_labels[i])) correct++;
    }
    std::cout << "\n\n######## Results ########" << std::endl;
    std::cout << "TOP_N used: " << TOP_N << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    std::cout << "Test accuracy: " << (100.0 * correct / test_features.size()) << "%" << std::endl;
    size_t tp = 0, tn = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < test_features.size(); ++i) {
        float prediction = nn.predict(test_features[i]);
        int pred_label = (prediction > 0.5f) ? 1 : 0;
        int true_label = static_cast<int>(test_labels[i]);
        if (pred_label == 1 && true_label == 1) tp++;
        else if (pred_label == 0 && true_label == 0) tn++;
        else if (pred_label == 1 && true_label == 0) fp++;
        else if (pred_label == 0 && true_label == 1) fn++;
    }
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "TP: " << tp << "  FP: " << fp << std::endl;
    std::cout << "FN: " << fn << "  TN: " << tn << std::endl;

    float precision = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0;
    float recall = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0;
    float f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "F1 Score: " << f1 << std::endl;
    return 0;
}