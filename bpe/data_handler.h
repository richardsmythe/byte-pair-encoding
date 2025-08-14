#pragma once
#include <vector>
#include <string>
#include "data.h"

class Data_Handler {
    std::vector<Data> data_array;
    std::vector<Data> training_data;
    std::vector<Data> test_data;
    std::vector<Data> validation_data;
    size_t spam_count = 0;
    size_t ham_count = 0;

public:
    Data_Handler();
    ~Data_Handler();

    void read_csv(const std::string& path, const std::string& delimiter = "\t");
    void split_data(float train_percent = 0.7f, float test_percent = 0.2f, float valid_percent = 0.1f);

    const std::vector<Data>& get_training_data() const;
    const std::vector<Data>& get_test_data() const;
    const std::vector<Data>& get_validation_data() const;
   
    size_t get_total_samples() const;

    std::vector<float> pad_or_truncate(const std::vector<uint32_t>& input, size_t fixed_size) const;
    
    void print_class_distribution() const;
    bool is_training_imbalanced(float threshold = 0.3f);
    void basic_smote();

};