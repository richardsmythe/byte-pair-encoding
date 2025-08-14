#include "data_handler.h"
#include "nn.h"
#include <iostream>
#include "bpe.h"
#include <sstream>

int main() {
    const size_t INPUT_SIZE = 64;
    const size_t ITERATIONS = 2000;
    const size_t SAMPLE_LIMIT = 2000;

    // load dataset and preprocess
    Data_Handler dh;
    dh.read_csv("SMSSpamCollection.txt", "\t");
    dh.split_data();

    // Check for class distribution in training set
    if (dh.is_training_imbalanced()) {
        std::cout << "Dataset is imbalanced. Rebalancing...\n";
        dh.basic_smote();
    }

    std::vector<std::vector<float>> train_features, test_features;
    std::vector<float> train_labels, test_labels;

    // prepare training data
    for (const auto& d : dh.get_training_data()) {
        train_features.push_back(dh.pad_or_truncate(d.get_feature_vector(), INPUT_SIZE));
        train_labels.push_back(static_cast<float>(d.get_label()));
    }

    // prepare test data
    for (const auto& d : dh.get_test_data()) {
        test_features.push_back(dh.pad_or_truncate(d.get_feature_vector(), INPUT_SIZE));
        test_labels.push_back(static_cast<float>(d.get_label()));
    }

    // train the NN
    NeuralNetwork nn(INPUT_SIZE, ITERATIONS);
    nn.train(train_features, train_labels, 0.1f);

    // Uncomment to write to file the BPE lookup table
   /* std::stringstream sample_text;
    size_t sample_count = 0;

    for (const auto& d : dh.get_training_data()) {
        for (auto token : d.get_feature_vector()) {
            sample_text << static_cast<char>(token);
        }
        sample_text << " ";
        if (++sample_count >= SAMPLE_LIMIT) break;
    }
    std::string vocab_sample = sample_text.str();
    bpe::PairArray vocab_pairs;
    bpe::Uint32Array vocab_tokens;
    bpe::run_bpe(vocab_sample, vocab_pairs, vocab_tokens);
    bpe::write_lookup_table("lookup_table.txt", vocab_pairs);
    std::cout << "Lookup table written to 'lookup_table.txt'\n";*/





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
    std::cout << "\n\nConfusion Matrix:\n";
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