#pragma once
#include "neural_net.h"
#include <string>

class LogicNeuralNetwork : public NeuralNet
{
public:
	LogicNeuralNetwork(const std::vector<size_t>& topology, const std::string& data_file_name, const std::string& network_name = "default");
	const std::string& GetDataFileName() const;
	const std::string& GetNetworkName() const;
	size_t GetTestingCorrect() const;
	size_t GetTestingIncorrect() const;
	void SetDataFileName(const std::string& data_file_name);
	void SetNetworkName(const std::string& network_name);
	bool TrainNetwork();

private:
	std::string data_file_name_;
	std::string network_name_;
	size_t testing_correct_;
	size_t testing_incorrect_;
	bool is_print_testing_info_; //TODO: add in this flag, which will be used in TrainNetwork();

};