#include "logic_neural_network.h"
#include <fstream>
#include <cassert>

LogicNeuralNetwork::LogicNeuralNetwork(const std::vector<size_t>& topology, const std::string& data_file_name, const std::string& network_name) : NeuralNet(topology), testing_correct_(0), testing_incorrect_(0), is_print_testing_info_(false)
{
	data_file_name_ = data_file_name;
	network_name_ = network_name;
}

const std::string& LogicNeuralNetwork::GetDataFileName() const
{
	return data_file_name_;
}

const std::string& LogicNeuralNetwork::GetNetworkName() const
{
	return network_name_;
}

size_t LogicNeuralNetwork::GetTestingCorrect() const
{
	return testing_correct_;
}

size_t LogicNeuralNetwork::GetTestingIncorrect() const
{
	return testing_incorrect_;
}
void LogicNeuralNetwork::SetDataFileName(const std::string& data_file_name)
{
	data_file_name_ = data_file_name;
}

void LogicNeuralNetwork::SetNetworkName(const std::string& network_name)
{
	network_name_ = network_name;
}

bool LogicNeuralNetwork::TrainNetwork()
{
	//defining and initializing variables
	std::vector<size_t>& topology = GetTopologyRef();
	std::vector<double> input_vals(topology.front());
	std::vector<double> target_vals(topology.back());
	std::vector<double> result_vals(topology.back());
	
	
	std::ifstream input(data_file_name_);
	if (!input.is_open()) return false;


	double read_data;

	//testing body loop
	while (input.good()) {

		//clear the vectors
		input_vals.clear();
		target_vals.clear();
		result_vals.clear();

		//getting the input data for one line of data
		for (size_t input_index = 0; input_index < topology.front(); ++input_index) {
			input >> read_data;
			input_vals.push_back(read_data);
		}

		//getting the output data for one line of data
		for (size_t output_index = 0; output_index < topology.back(); ++output_index) {
			input >> read_data;
			target_vals.push_back(read_data);
		}

		
		
		//TODO: add in a flag for printing the testing data while it is ran
		//if(is_print_testing_info_)

		FeedForward(input_vals);
		BackPropagate(target_vals);
		result_vals = GetResults();
		result_vals.front() = (result_vals[0] < 0.5) ? 0 : 1;

		if (result_vals.front() == target_vals.front()) {
			++testing_correct_;
		}
		else {
			++testing_incorrect_;
		}

		//making sure the training data inputs match topology quantities:
		assert(input_vals.size() == topology.front());
		assert(target_vals.size() == topology.back());
	}

	return true;
}