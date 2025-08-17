#include "data_handler.h"
#include "nn.h"
#include <iostream>
#include "bpe.h"
#include <sstream>
#include <algorithm>

int main() {
    const size_t INPUT_SIZE = 32;
    const size_t ITERATIONS = 4000;
    const size_t TOP_N = 100;

    // load dataset and preprocess
    Data_Handler dh;
    dh.read_csv("SMSSpamCollection.txt", "\t");
    dh.split_data(); 

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

    // calculate class weights for weighted loss
    // weights for each class is inversely proportional to its frequency
    float weight_ham = 1.0f, weight_spam = 8.0f; // higher for spam as it's minority class
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