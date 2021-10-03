#pragma once
#include "neural_net.h"
#include <string>

class LogicNeuralNetwork : public NeuralNet
{
public:
	LogicNeuralNetwork(const std::vector<size_t>& topology, const std::string& data_file_name, const std::string& network_name="default");
	const std::string& GetDataFileName() const;
	void SetDataFileName(const std::string& data_file_name);
	const std::string& GetNetworkName() const;
	void SetNetworkName(const std::string& network_name);

private:
	void TrainNetwork();
	std::string data_file_name_;
	std::string network_name_;
	
};