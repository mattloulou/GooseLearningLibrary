#include "preset_network_test.h"
#include "logic_neural_network.h"
#include <numeric>
#include <iostream>

void PresetNetworkTests::LogicNetworkTest()
{

	//creating vector of networks to test
	std::vector<LogicNeuralNetwork> logic_neural_networks; //TODO: should this vector type have &?
	std::vector<double> accuracies;
	double average_accuracy;
	

	//adding sample networks to the vector
	logic_neural_networks.push_back(LogicNeuralNetwork({ 2,2,1 }, "xorTrainingData.txt", "xor"));
	logic_neural_networks.push_back(LogicNeuralNetwork({ 4,4,2,1 }, "doubleImplicationTrainingData.txt", "double implication"));
	size_t network_size = logic_neural_networks.size();

	for (size_t network_index = 0; network_index < network_size; ++network_index) {
		LogicNeuralNetwork& network = logic_neural_networks[network_index];
		network.TrainNetwork();
		accuracies.push_back(double(network.GetTestingCorrect()) / 
								(double(network.GetTestingCorrect()) + double(network.GetTestingIncorrect())));
	}

	//average accuracy is of range [0.0,1.0]
	average_accuracy = std::accumulate(accuracies.begin(), accuracies.end(), 0.0) / double(network_size);

	//printing out the result of the tests:
	std::cout << "Average accuracy in " << network_size;
	if (network_size != 1) { //if structure for plurality
		std::cout << " tests  was ";
	}
	else {
		std::cout << " test  was ";
	}
	std::cout << average_accuracy * 100 << "%." << std::endl;
}