#include "preset_network_test.h"
#include "logic_neural_network.h"

void PresetNetworkTests::LogicNetworkTest() const
{


	//creating vector of networks to test
	std::vector<LogicNeuralNetwork> logic_neural_networks; //TODO: should this vector type have &?
	std::vector<double> accuracies;
	double total_accuracy;

	//adding sample networks to the vector
	logic_neural_networks.push_back(LogicNeuralNetwork({ 2,2,1 }, "xorTrainingData.txt", "xor"));
	logic_neural_networks.push_back(LogicNeuralNetwork({ 4,4,2,1 }, "doubleImplicationTrainingData.txt", "double implication"));

	for (size_t network_index = 0; network_index < logic_neural_networks.size(); ++network_index) {
		LogicNeuralNetwork& network = logic_neural_networks[network_index];
		network.TrainNetwork();
		accuracies.push_back(double(network.GetTestingCorrect()) / double(network.GetTestingIncorrect()));
	}

	total_accuracy = std::accumulate(accuracies.begin(), accuracies.end(),);

}