#pragma once
#include "connection.h"
#include <vector>

class Neuron;

using Layer = std::vector<Neuron>;

class Neuron 
{
public:
    Neuron(size_t num_outputs, size_t my_index);
    double GetOutputVal(void) const;
    void SetOutputVal(double val);
    void FeedFrom(const Layer &prev_layer);
    void CalcOutputGradients(double target_val);
    void CalcHiddenGradients(Layer &next_layer);
    void UpdateInputWeights(Layer &prev_layer);
    std::vector<Connection>& GetOutputWeightsRef();
    const std::vector<Connection>& GetOutputWeightsRef() const;

private:
    size_t my_index_;
    double output_val_;
    double gradient_;
    std::vector<Connection> output_weights_;


    static double TransferFunction(double x);
    static double TransferFunctionDerivative(double x);
    static double RandomWeight(void);
    static constexpr double eta_ = 0.15; //domain [0.0, 1.0] overall net training rate
    static constexpr double alpha_ = 0.3; //domain [0.0, n] multiplier of last weight change (momentum) (can be above 1 it appears, as it is from [0,n], not [0,1], according to the tutorial)

};
