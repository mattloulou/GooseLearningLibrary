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
	std::ifstream input(data_file_name_);
	if (!input.is_open()) return false;

	std::string dataLine;
	while (std::getline(input, dataLine)) {

	}
	
	//defining variables
	/*std::vector<double> input_vals(topology.front());
	std::vector<double> target_vals(topology.back());
	std::vector<double> result_vals(topology.back());*/
}