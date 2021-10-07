#include "preset_network_test.h"
#include "logic_neural_network.h"

void PresetNetworkTests::LogicNetworkTest() const
{
	//creating vector of networks to test
	std::vector<LogicNeuralNetwork> logic_neural_networks;

	//adding sample networks to the vector
	logic_neural_networks.push_back(LogicNeuralNetwork({ 2,2,1 }, "xorTrainingData.txt", "xor"));
	logic_neural_networks.push_back(LogicNeuralNetwork({ 4,4,2,1 }, "doubleImplicationTrainingData.txt", "double implication"));

	for (size_t network_index = 0; network_index < logic_neural_networks.size(); ++network_index) {

	}


}