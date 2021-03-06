#include "neural_net.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cassert>

NeuralNet::NeuralNet(const std::vector<size_t> &topology) : error_(0.0), recent_average_error_(0.0), recent_average_smoothing_factor_(0.0), topology_(topology)
{
    for (size_t layer_index = 0; layer_index < topology.size(); ++layer_index) 
    {
        this->layers_.push_back(Layer());

        //get the number of output neuron for the next layer
        size_t num_outputs = (layer_index != topology.size() - 1) ? topology[layer_index + 1] : 0;
    
        //fill the new layer with neruons, and
        //an extra bias neuron (hence <= instead of <)
        for (size_t neuron_index = 0; neuron_index <= topology[layer_index]; ++neuron_index) 
        {
            this->layers_.back().push_back(Neuron(num_outputs, neuron_index));
        }

        //set bias neuron's value
        this->layers_.back().back().SetOutputVal(1.0);
    }
}

void NeuralNet::FeedForward(const std::vector<double> &input_vals) 
{


    //validate the input data size, -1 to filter the hidden bias
    assert(input_vals.size() == this->layers_.front().size() - 1);

    //copy the input data into the input layer
    for (size_t neuron_index = 0; neuron_index < input_vals.size(); ++neuron_index) 
    {
        this->layers_.front()[neuron_index].SetOutputVal(input_vals[neuron_index]);
    }

    //forward propogate
    for (size_t layer_index = 1; layer_index < this->layers_.size(); ++layer_index) 
    {
        Layer &curr_layer = this->layers_[layer_index];
        const Layer &prev_layer = this->layers_[layer_index - 1];

        //-1 to filter the bias neuron
        for (size_t neuron_index = 0; neuron_index < this->layers_[layer_index].size() - 1; ++neuron_index) 
        {
           curr_layer[neuron_index].FeedFrom(prev_layer);
        }
    }
}


void NeuralNet::BackPropagate(const std::vector<double> &target_vals)
{
    // Calculate mean square root error (RMS) for output neurons
    Layer &output_layer  = this->layers_.back();
    this->error_ = 0.0;

    //-1 filter bias neuron
    for (size_t neuron_index = 0; neuron_index < output_layer.size() - 1; ++neuron_index) 
    { 
        double delta = target_vals[neuron_index] - output_layer[neuron_index].GetOutputVal();
        this->error_ += delta * delta;
    }

    this->error_ /= double(output_layer.size() - 1); //get average error squared, -1 filter bias neuron
    this->error_ = sqrt(this->error_);

    //calculate recent average error performance
    this->recent_average_error_ = (this->recent_average_error_ * this->recent_average_smoothing_factor_ + this->error_) / (this->recent_average_smoothing_factor_ + 1.0);


    //calculate output layer gradients, -1 filter bias neuron
    for (size_t neuron_index = 0; neuron_index < output_layer.size() - 1; ++neuron_index)
     {
        output_layer[neuron_index].CalcOutputGradients(target_vals[neuron_index]);
    }


    //calculate gradients on hidden layers
    for (size_t layer_index = this->layers_.size() - 2; layer_index > 0; --layer_index) 
    {
        Layer &curr_layer = this->layers_[layer_index];
        Layer &next_layer = this->layers_[layer_index + 1];

        for(Neuron& neuron : curr_layer)
        {
            neuron.CalcHiddenGradients(next_layer);
        }
    }

    //update the weights for all layers except the output layer
    for (size_t layer_index = this->layers_.size() - 1; layer_index > 0; --layer_index) 
    {
        Layer &curr_layer = this->layers_[layer_index];
        Layer &prev_layer = this->layers_[layer_index -1];

        //-1 filter bias neuron
        for (size_t neuron_index = 0; neuron_index < curr_layer.size() - 1; ++neuron_index) 
        {
            curr_layer[neuron_index].UpdateInputWeights(prev_layer);
        }
    }

}

std::vector<double> NeuralNet::GetResults() const
{
    //return through std::move() implicitly, faster than reference
    std::vector<double> result_vals;

    //output layer
    const Layer& output_layer = this->layers_.back();

    //-1 filter bias neuron
    for (size_t neuron_index = 0; neuron_index < output_layer.size() - 1; ++neuron_index)
    { 
        result_vals.push_back(output_layer[neuron_index].GetOutputVal());
    }

    return result_vals;
}

double NeuralNet::GetError() const 
{
     return this->error_; 
}

std::vector<Layer>& NeuralNet::GetLayerRef()
{
    return this->layers_;
}
const std::vector<Layer>& NeuralNet::GetLayerRef() const
{
    return this->layers_;
}
std::vector<size_t>& NeuralNet::GetTopologyRef()
{
    return this->topology_;
}
const std::vector<size_t>& NeuralNet::GetTopologyRef() const
{
    return this->topology_;
}



bool NeuralNet::ExportNetworkFile(const std::string& file_name)
{
    if (this->layers_.empty()) return false;

    std::ofstream output(file_name, std::ofstream::trunc | std::ofstream::binary);

    //save topology structure
    size_t layer_size = this->layers_.size();
    output.write(reinterpret_cast<const char*>(&layer_size), sizeof(size_t));
    std::clog << "layer size: " << layer_size << '\n';

    for (const Layer& layer : this->layers_)
    {
        size_t neuron_size = layer.size();
        output.write(reinterpret_cast<const char*>(&neuron_size), sizeof(size_t));
        std::clog << "neuron size: " << neuron_size << '\n';
    }

    
    //save neuron output weights
    for (const Layer& layer : this->layers_)
    {
        for (const Neuron& neuron : layer)
        {
            const std::vector<Connection>& output_weights = neuron.GetOutputWeightsRef();
            output.write(reinterpret_cast<const char*>(output_weights.data()), output_weights.size() * sizeof(Connection));
        }
    }
    
    output.close();
    return true;
}
bool NeuralNet::ImportNetworkFile(const std::string& file_name)
{
    std::ifstream input(file_name, std::ifstream::binary);
    if (!input.is_open()) return false;

    //read topology structure
    size_t layer_size = 0;
    input.read(reinterpret_cast<char*>(&layer_size), sizeof(size_t));

    std::cout << "Layer size: " << layer_size << '\n';


    std::vector<size_t> topology(layer_size);
    input.read(reinterpret_cast<char*>(topology.data()), layer_size * sizeof(size_t));

    for (auto val : topology)
    {
        std::cout << val << '\n';
    }


    //reconstruct the neural network layout
    this->layers_.clear();
    for (size_t layer_index = 0; layer_index < topology.size(); ++layer_index)
    {
        this->layers_.push_back(Layer());

        size_t num_outputs = (layer_index != topology.size() - 1) ? topology[layer_index + 1] : 0;

        // < instead of <=, because the topology already contains the bias neuron
        for (size_t neuron_index = 0; neuron_index < topology[layer_index]; ++neuron_index)
        {
            this->layers_.back().push_back(Neuron(num_outputs, neuron_index));
        }

        this->layers_.back().back().SetOutputVal(1.0);
    }

    
    //read neuron output weights
    for (Layer& layer : this->layers_)
    {
        for (Neuron& neuron : layer)
        {
            std::vector<Connection>& output_weights = neuron.GetOutputWeightsRef();
            input.read(reinterpret_cast<char*>(output_weights.data()), output_weights.size() * sizeof(Connection));
        }
    }
    
    input.close();
    return true;
}

void NeuralNet::PrintLayerInfo() const
{
    int i = 0;
    for (const Layer& layer : this->layers_)
    {
        std::clog << "Layer: " << ++i << '\n';
        for (const Neuron& neuron : layer)
        {
            const std::vector<Connection>& output_weights = neuron.GetOutputWeightsRef();
            for (auto& val : output_weights)
            {
                std::cout << val.weight_ << ' ';
            }
            std::cout << '\n';
        }
    }
}