#pragma once
#include "neuron.h"
#include <vector>
#include <string>

class NeuralNet
{
public:
    NeuralNet(const std::vector<size_t> &topology);
    void FeedForward(const std::vector<double> &input_vals);
    void BackPropagate(const std::vector<double> &target_vals);
    std::vector<double> GetResults() const;
    double GetError() const;
    std::vector<Layer>& GetLayerRef();
    const std::vector<Layer>& GetLayerRef() const;
    std::vector<size_t>& GetTopologyRef();
    const std::vector<size_t>& GetTopologyRef() const;

    bool ExportNetworkFile(const std::string& file_name);
    bool ImportNetworkFile(const std::string& file_name);
    void PrintLayerInfo() const;

private:
    std::vector<Layer> layers_;
    std::vector<size_t> topology_;
    double error_;
    double recent_average_error_;
    double recent_average_smoothing_factor_;
};