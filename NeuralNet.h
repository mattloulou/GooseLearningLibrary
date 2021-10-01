#pragma once
#include "Neuron.h"
#include <vector>

class NeuralNet
{
public:
    NeuralNet(const std::vector<size_t> &topology);
    void FeedForward(const std::vector<double> &input_vals);
    void BackPropagate(const std::vector<double> &target_vals);
    std::vector<double> GetResults() const;
    double GetError() const;
private:
    std::vector<Layer> layers_;
    double error_;
    double recent_average_error_;
    double recent_average_smoothing_factor_;
};