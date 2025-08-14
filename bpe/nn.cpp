#include "nn.h"
#include <random>
#include <cmath>
#include <iostream>

NeuralNetwork::NeuralNetwork(size_t input_size, size_t iterations)
    : input_size(input_size), iterations(iterations) {
    w_hidden_1.resize(input_size);
    w_hidden_2.resize(input_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < input_size; ++i) {
        w_hidden_1[i] = dis(gen);
        w_hidden_2[i] = dis(gen);
    }
    b_hidden_1 = dis(gen);
    b_hidden_2 = dis(gen);
    w_h_output_1 = dis(gen);
    w_h_output_2 = dis(gen);
    b_output = dis(gen);
}

float NeuralNetwork::sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}
float NeuralNetwork::sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

float NeuralNetwork::forward(const std::vector<float>& x) {
    h1_input = b_hidden_1;
    h2_input = b_hidden_2;
    for (size_t i = 0; i < input_size; ++i) {
        h1_input += x[i] * w_hidden_1[i];
        h2_input += x[i] * w_hidden_2[i];
    }
    h1_output = sigmoid(h1_input);
    h2_output = sigmoid(h2_input);
    out_input = h1_output * w_h_output_1 + h2_output * w_h_output_2 + b_output;
    y_pred = sigmoid(out_input);
    return y_pred;
}

void NeuralNetwork::backward(const std::vector<float>& x, float y_true, float magnitude) {
    float d_loss_d_ypred = 2 * (y_pred - y_true);
    float d_ypred_d_out_input = sigmoid_derivative(out_input);
    float d_loss_d_out_input = d_loss_d_ypred * d_ypred_d_out_input;

    float grad_w_h_output_1 = d_loss_d_out_input * h1_output;
    float grad_w_h_output_2 = d_loss_d_out_input * h2_output;
    float grad_b_output = d_loss_d_out_input;

    float d_loss_d_h1_output = d_loss_d_out_input * w_h_output_1;
    float d_loss_d_h2_output = d_loss_d_out_input * w_h_output_2;
    float d_h1_output_d_h1_input = sigmoid_derivative(h1_input);
    float d_h2_output_d_h2_input = sigmoid_derivative(h2_input);
    float d_loss_d_h1_input = d_loss_d_h1_output * d_h1_output_d_h1_input;
    float d_loss_d_h2_input = d_loss_d_h2_output * d_h2_output_d_h2_input;

    for (size_t i = 0; i < input_size; ++i) {
        float grad_w1_hidden_1 = d_loss_d_h1_input * x[i];
        float grad_w1_hidden_2 = d_loss_d_h2_input * x[i];
        w_hidden_1[i] -= magnitude * grad_w1_hidden_1;
        w_hidden_2[i] -= magnitude * grad_w1_hidden_2;
    }
    b_hidden_1 -= magnitude * d_loss_d_h1_input;
    b_hidden_2 -= magnitude * d_loss_d_h2_input;
    w_h_output_1 -= magnitude * grad_w_h_output_1;
    w_h_output_2 -= magnitude * grad_w_h_output_2;
    b_output -= magnitude * grad_b_output;
}

float NeuralNetwork::train(const std::vector<std::vector<float>>& X, const std::vector<float>& y, float magnitude) {
    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < X.size(); ++i) {
            forward(X[i]);
            backward(X[i], y[i], magnitude);
        }
    }
   
    return 0.0f;
}

float NeuralNetwork::predict(const std::vector<float>& x) {
    return forward(x);
}

