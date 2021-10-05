#include "logic_neural_network.h"
#include <fstream>

LogicNeuralNetwork::LogicNeuralNetwork(const std::vector<size_t>& topology, const std::string& data_file_name, const std::string& network_name) : NeuralNet(topology)
{
	data_file_name_ = data_file_name;
	network_name_ = network_name;

}

const std::string& LogicNeuralNetwork::GetDataFileName() const
{
	return data_file_name_;
}

void LogicNeuralNetwork::SetDataFileName(const std::string& data_file_name)
{
	data_file_name_ = data_file_name;
}

const std::string& LogicNeuralNetwork::GetNetworkName() const
{
	return network_name_;
}

void LogicNeuralNetwork::SetNetworkName(const std::string& network_name)
{
	network_name_ = network_name;
}

bool LogicNeuralNetwork::TrainNetwork()
{

	std::vector<size_t>& topology = GetTopologyRef();
	//defining variables
	std::vector<double> input_vals(topology.front());
	std::vector<double> target_vals(topology.back());
	std::vector<double> result_vals(topology.back());
	
	
	std::ifstream input(data_file_name_);
	if (!input.is_open()) return false;

	double read_data;

	while (input.good()) {


		for (size_t input_index = 0; input_index < topology.front(); ++input_index) {
			input >> read_data;
			input_vals.push_back(read_data);
		}

		//TODO: need loop for last layer
		for (size_t output_index = 0; output_index < topology.back(); ++output_index) {
			input >> read_data;
			input_vals.push_back(read_data);
		}
	}
	
}
