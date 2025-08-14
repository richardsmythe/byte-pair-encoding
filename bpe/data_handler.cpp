#include "data_handler.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <direct.h>
#include <fstream>
#include <string>
#include "bpe.h"
#include <random>


Data_Handler::Data_Handler() : spam_count(0), ham_count(0) {}
Data_Handler::~Data_Handler() {}

/// <summary>
/// Reads a CSV file containing SMS messages and labels, tokenizes each message using BPE,
/// and stores the resulting feature vectors and labels in the data_array.
/// Each line should be in the format: label<TAB>message, where label is 'ham' or 'spam'.
/// The feature vector for each message is a vector of BPE token IDs.
/// </summary>
void Data_Handler::read_csv(const std::string& path, const std::string& delimiter) {
	std::ifstream data_file(path.c_str());
	std::string line;
	while (std::getline(data_file, line)) {
		if (line.empty()) continue;

		size_t tab_pos = line.find(delimiter);

		if (tab_pos == std::string::npos) continue;

		std::string label_str = line.substr(0, tab_pos);
		std::string text = line.substr(tab_pos + delimiter.length());
		uint8_t label = (label_str == "spam") ? 1 : 0;
		Data d;

		std::vector<uint32_t>tokens;
		bpe::PairArray pairs;
		bpe::Uint32Array tokens_out;
		bpe::run_bpe(text, pairs, tokens_out);
		tokens.assign(tokens_out.begin(), tokens_out.end());
		d.set_feature_vector(tokens);

		d.set_label(label);
		data_array.push_back(d);
	}
}

/// <summary>
///  Randomly divide the loaded dataset (data_array) into three separate subsets: training_data, test_data, and validation_data. 
/// </summary>
void Data_Handler::split_data(float train_percent, float test_percent, float valid_percent) {
	std::vector<int> indices(data_array.size());
	for (int i = 0; i < data_array.size(); ++i) {
		indices[i] = i;
	}
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(indices.begin(), indices.end(), g);

	int train_size = static_cast<int>(data_array.size() * train_percent);
	int test_size = static_cast<int>(data_array.size() * test_percent);
	int valid_size = static_cast<int>(data_array.size() * valid_percent);

	for (int i = 0; i < train_size; ++i) {
		training_data.push_back(data_array[indices[i]]);
	}
	for (int i = train_size; i < train_size + test_size; ++i) {
		test_data.push_back(data_array[indices[i]]);
	}
	for (int i = train_size + test_size; i < train_size + test_size + valid_size; ++i) {
		validation_data.push_back(data_array[indices[i]]);
	}

	std::cout << "Training data size: " << training_data.size() << std::endl;
	std::cout << "Test data size: " << test_data.size() << std::endl;
	std::cout << "Validation data size: " << validation_data.size() << std::endl;
}

/// <summary>
/// Prints the class distribution (ham/spam counts) in the training set.
/// </summary>
void Data_Handler::print_class_distribution() const {
    std::cout << "Training set: ham = " << ham_count << ", spam = " << spam_count << std::endl;
}

/// <summary>
/// Checks if the training set is imbalanced based on a given threshold.
/// </summary>
bool Data_Handler::is_training_imbalanced(float threshold) {
    spam_count = 0;
    ham_count = 0;
    for (const auto& d : training_data) {
        if (d.get_label() == 1) spam_count++;
        else ham_count++;
    }
    std::cout << "Training set: ham = " << ham_count << ", spam = " << spam_count << std::endl;
    size_t minority = std::min(spam_count, ham_count);
    size_t majority = std::max(spam_count, ham_count);
    return (minority < threshold * majority);
}

/// <summary>
///  Makes feature vectors a fixed size for the NN input
/// </summary>
std::vector<float> Data_Handler::pad_or_truncate(const std::vector<uint32_t>& input, size_t fixed_size) const {
	std::vector<float> result(fixed_size, 0.0f); // Initialize with zeros  
	size_t copy_size = std::min(input.size(), fixed_size);
	for (size_t i = 0; i < copy_size; ++i) {
		result[i] = static_cast<float>(input[i]);
	}
	return result;
}

/// <summary>
/// Basic SMOTE to interpolate between 2 random spam samples and generate synthetic data for balancing.
/// For every synthetic sample needed it randomly selects 2 samples, pads, creates new Data object 
/// with this synthetic feature vexctor and sets label as 1 for spam.
/// </summary>
void Data_Handler::basic_smote() {
	std::vector<Data> spam_samples;
	// get all spam samples
	for (const auto& d : training_data) {
		if (d.get_label() == 1) spam_samples.push_back(d);
	}

	// size_t needed = ham_count - spam_count;
	// less aggressive - will only generate enough synthetic spam samples to bring the spam count up to 50% of the ham count
	size_t target_spam = static_cast<size_t>(0.5 * ham_count); // 50% of ham
	size_t needed = (spam_count < target_spam) ? (target_spam - spam_count) : 0;
	
	if (spam_samples.empty() || needed == 0) return;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, spam_samples.size() - 1);

	for (size_t i = 0; i < needed; ++i) {
		// Pick two random spam samples
		const Data& random_spam_sample1 = spam_samples[dis(gen)];
		const Data& random_spam_sample2 = spam_samples[dis(gen)];
		std::vector<uint32_t> new_features;
		const auto& f1 = random_spam_sample1.get_feature_vector();
		const auto& f2 = random_spam_sample2.get_feature_vector();
		size_t len = std::min(f1.size(), f2.size());
		for (size_t j = 0; j < len; ++j) {
			// randomly pick a token from random_spam_sample1 or random_spam_sample2
			new_features.push_back((gen() % 2 == 0) ? f1[j] : f2[j]);
		}
		// pads new synthetic feature vector
		while (new_features.size() < f1.size()) new_features.push_back(0);
		Data synthetic;
		synthetic.set_feature_vector(new_features);
		synthetic.set_label(1); 
		training_data.push_back(synthetic);
	}
}

const std::vector<Data>& Data_Handler::get_training_data() const {
	return training_data;
}
const std::vector<Data>& Data_Handler::get_test_data() const {
	return test_data;
}
const std::vector<Data>& Data_Handler::get_validation_data() const {
	return validation_data;
}

size_t Data_Handler::get_total_samples() const {
	return data_array.size();
}
