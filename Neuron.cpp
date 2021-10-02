#include "neuron.h"

Neuron::Neuron(size_t num_outputs, size_t my_index) : output_val_(0), output_weights_(), my_index_(0), gradient_(0.0)
{
    for (size_t connection_index = 0; connection_index < num_outputs; ++connection_index) {
        output_weights_.push_back(Connection());
        output_weights_.back().weight_ = RandomWeight();
    }
    my_index_ = my_index;
}

double Neuron::GetOutputVal(void) const
{
    return output_val_;
}

void Neuron::SetOutputVal(double val) 
{
    output_val_ = val;
}

void Neuron::FeedFrom(const Layer &prev_layer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    for (size_t neuron_index = 0; neuron_index < prev_layer.size(); ++neuron_index) {
        sum += prev_layer[neuron_index].output_val_ * 
        prev_layer[neuron_index].output_weights_[my_index_].weight_;
    }

    output_val_ = Neuron::TransferFunction(sum);
}

void Neuron::CalcOutputGradients(double target_val)
{
    double delta = target_val - output_val_;
    gradient_ = delta* Neuron::TransferFunctionDerivative(output_val_);
}

void Neuron::CalcHiddenGradients(Layer &next_layer)
{
    double dow = 0.0;

    // Sum our contributions of the errors at the nodes we feed
    for (size_t neuron_index = 0; neuron_index < next_layer.size() - 1 /*no bias neuron*/; ++neuron_index) {
        dow += output_weights_[neuron_index].weight_ * next_layer[neuron_index].gradient_;
    }

    this->gradient_ = dow * Neuron::TransferFunctionDerivative(output_val_);
}

void Neuron::UpdateInputWeights(Layer &prev_layer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (size_t neuron_index = 0; neuron_index < prev_layer.size(); ++neuron_index) {
        Neuron &neuron = prev_layer[neuron_index];
        double old_delta_weight = neuron.output_weights_[my_index_].delta_weight_;

        double new_delta_weight =
                //individual input, mangified by the gradient and train rate:
                eta_ // overall net learning rate "pronounced: ey-ta"

                //eta_: 0.0 - slow learner; 0.2 - medium learner; 1.0 - reckless learner

                * neuron.GetOutputVal()
                * gradient_
                // Also add momentum = a fraction of the previus delta weight
                + alpha_ //momentum. 0.0 - no momentum; 0.5 - moderate momentum
                * old_delta_weight;

            neuron.output_weights_[my_index_].delta_weight_ = new_delta_weight;
            neuron.output_weights_[my_index_].weight_ += new_delta_weight;
    }
}

std::vector<Connection>& Neuron::GetOutputWeightsRef()
{
    return this->output_weights_;
}

const std::vector<Connection>& Neuron::GetOutputWeightsRef() const
{
    return this->output_weights_;
}





double Neuron::TransferFunction(double x)
{
    // tanh - output range (-1, 1)
    return tanh(x);
}

double Neuron::TransferFunctionDerivative(double x)
{
    // there exists a tanh derivative approximation of 1.0 - x*x
    // I found a different approximation which looks more accurate, which is 1.0/(x*x + 1)
    // the real 100% derviative is 1.0 - (tanh(x)) * (tanh(x))
    double tanhx = tanh(x);
    return 1.0 - tanhx*tanhx;
}

double Neuron::RandomWeight(void)
{
    return double(rand()) / double(RAND_MAX);
}