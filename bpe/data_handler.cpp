#include "data_handler.h"
#include <iostream>
#include <set>
#include <algorithm>
#include <direct.h>
#include <fstream>
#include <string>
#include "bpe.h"
#include <random>
#include <unordered_map>


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
	count_ham_spam(training_data, ham_count, spam_count);
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
/// Turn the tokens in to an embedding vector of more meaningful data for the NN model.
/// Averaging the embeddings summarizes the message in a way the NN can better udnerstand.
/// The embedding size can change but for now it's fine at 32.
/// </summary>
std::vector<float> Data_Handler::embed_and_average(const std::vector<uint32_t>& input, size_t embedding_size) const {
	static std::unordered_map<uint32_t, std::vector<float>> embedding_matrix;
	static bool initialized = false;
	if (!initialized) {
		// create embedding matrix with random vectors for each token ID
		uint32_t max_token = 0;
		for (const auto& d : data_array) {
			for (auto t : d.get_feature_vector()) {
				if (t > max_token) max_token = t;
			}
		}
		std::mt19937 gen(42);
		std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
		for (uint32_t i = 0; i <= max_token; ++i) {
			std::vector<float> emb(embedding_size);
			for (size_t j = 0; j < embedding_size; ++j) emb[j] = dis(gen);
			embedding_matrix[i] = emb;
		}
		initialized = true;
	}

	// look up embeddings for each token in the message
	std::vector<float> result(embedding_size, 0.0f);
	if (input.empty()) return result;
	for (auto t : input) {
		const auto& emb = embedding_matrix[t];
		for (size_t j = 0; j < embedding_size; ++j) result[j] += emb[j];
	}
	// now average the embeddings to get a single vector for the message
	for (size_t j = 0; j < embedding_size; ++j) result[j] /= input.size();
	return result;
}

/// <summary>
/// Calculate the cosine similarity between two embedding vectors.
/// Returns a value between -1 and 1, where 1 means identical vectors,
/// 0 means orthogonal vectors, and -1 means opposite vectors.
/// </summary>
float Data_Handler::cosine_similarity(const std::vector<float>& vec1, const std::vector<float>& vec2) const {
    if (vec1.size() != vec2.size() || vec1.empty()) {
        return 0.0f; // Invalid vectors
    }

    float dot_product = 0.0f;
    float magnitude1 = 0.0f;
    float magnitude2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    magnitude1 = std::sqrt(magnitude1);
    magnitude2 = std::sqrt(magnitude2);

    if (magnitude1 < 1e-10 || magnitude2 < 1e-10) {
        return 0.0f; // Avoid division by zero
    }

    return dot_product / (magnitude1 * magnitude2);
}

/// <summary>
/// Helps distinguish between spam and ham by identifying features (words/tokens) that appear
/// much more frequently in one class than the other. For each feature, it measures the difference
/// between observed and expected occurrences in spam and ham, selecting those with the largest differences
/// </summary>
std::vector<uint32_t> Data_Handler::select_features_chi_square(size_t top_n) const {
	// chi_sqr = sum observed-expected)^2 / expected
	size_t ham_total = 0, spam_total = 0;
	count_ham_spam(training_data, ham_total, spam_total);

	std::unordered_map<uint32_t, size_t> spam_with_feature;
	std::unordered_map<uint32_t, size_t> ham_with_feature;
	std::set<uint32_t> all_features_seen;


	// track unique features for each ham and spam
	for (const auto& message : training_data) {
		std::set<uint32_t> unique_features_in_message(message.get_feature_vector().begin(), message.get_feature_vector().end());
		for (auto feature : unique_features_in_message) {
			if (message.get_label() == 1) {
				spam_with_feature[feature]++;
			}
			else {
				ham_with_feature[feature]++;
			}
			all_features_seen.insert(feature);
		}
	}

	// calculate chi-square for each feature
	struct FeatureScore {
		uint32_t feature;
		double chi_sqr;
		bool operator<(const FeatureScore& other) const {
			return chi_sqr > other.chi_sqr; 
		}
	};

	std::vector<FeatureScore> scores;
	for (const auto& feature : all_features_seen) {
		size_t A = spam_with_feature[feature];
		size_t B = ham_with_feature[feature];
		size_t C = spam_total - A;
		size_t D = ham_total - B;

		double total = spam_total + ham_total;
		double expected_A = (A + B) * spam_total / total;
		double expected_B = (A + B) * ham_total / total;
		double expected_C = (C + D) * spam_total / total;
		double expected_D = (C + D) * ham_total / total;

		double chi_sqr = 0.0; 
		if (expected_A > 0) chi_sqr += ((A - expected_A) * (A - expected_A)) / expected_A;
		if (expected_B > 0) chi_sqr += ((B - expected_B) * (B - expected_B)) / expected_B;
		if (expected_C > 0) chi_sqr += ((C - expected_C) * (C - expected_C)) / expected_C;
		if (expected_D > 0) chi_sqr += ((D - expected_D) * (D - expected_D)) / expected_D;

		scores.push_back({ feature, chi_sqr });
	}	

	std::sort(scores.begin(), scores.end());

	std::vector<uint32_t> top_features;
	for (size_t i = 0; i < std::min(top_n, scores.size()); i++)
	{
		top_features.push_back(scores[i].feature);
	}

	return top_features;
}



/// <summary>
/// Counts the number of ham and spam samples in the given data vector.
/// </summary>
void Data_Handler::count_ham_spam(const std::vector<Data>& data, size_t& ham_count, size_t& spam_count) const {
	ham_count = 0;
	spam_count = 0;
	for (const auto& d : data) {
		if (d.get_label() == 1) spam_count++;
		else ham_count++;
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

size_t Data_Handler::get_vocabulary_size() const {
	std::set<uint32_t> unique_tokens;
	for (const auto& d : data_array) {
		for (auto token : d.get_feature_vector()) {
			unique_tokens.insert(token);
		}
	}
	return unique_tokens.size();
}