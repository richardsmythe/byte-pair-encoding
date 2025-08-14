#include "data.h"
#include <iostream>

Data::Data() {}   
Data::~Data() {}

void Data::set_feature_vector(const std::vector<uint32_t>& vect) {
    feature_vector = vect;
}

void Data::append_to_feature_vector(uint32_t val) {
    feature_vector.push_back(val);
}

void Data::set_label(uint8_t val) {
    label = val;
}

const std::vector<uint32_t>& Data::get_feature_vector() const {
    return feature_vector;
}

uint8_t Data::get_label() const {
    return label;
}

