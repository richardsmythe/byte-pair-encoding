#pragma once
#include <vector>
#include <cstddef>

class NeuralNetwork {
public:
    NeuralNetwork(size_t input_size, size_t iterations);
    float train(const std::vector<std::vector<float>>& X, const std::vector<float>& y, float magnitude, float weight_ham, float weight_spam);
    float predict(const std::vector<float>& x);
private:
    float sigmoid(float x);
    float sigmoid_derivative(float x);
    float forward(const std::vector<float>& x);
    void backward(const std::vector<float>& x, float y_true, float magnitude);
    size_t input_size;
    size_t iterations;
    std::vector<float> w_hidden_1, w_hidden_2;
    float b_hidden_1, b_hidden_2;
    float w_h_output_1, w_h_output_2, b_output;
    float h1_input, h1_output, h2_input, h2_output, out_input, y_pred;
};
