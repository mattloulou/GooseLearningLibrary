#include "neural_net.h"
#include "preset_network_test.h"
#include <iostream>

using namespace std;

void xorTest()
{
    //creating the topology for the network
    std::vector<size_t> topology;
    topology = { 2,2,1 };

    NeuralNet my_net(topology);

    //defining variables
    std::vector<double> input_vals = { 0, 0 };
    std::vector<double> target_vals = { 0 };
    std::vector<double> result_vals = { 0 };

    //doing the training
    for (int i = 1; i <= 1000; ++i) {
        std::cin >> input_vals[0];
        std::cin >> input_vals[1];
        std::cin >> target_vals[0];
        std::cout << "Inputs #" << i << ": " << input_vals[0] << " " << input_vals[1] << std::endl;
        std::cout << "Target Value: " << target_vals[0] << std::endl;

        my_net.FeedForward(input_vals);
        my_net.BackPropagate(target_vals);
        result_vals = my_net.GetResults();

        result_vals[0] = (result_vals[0] < 0.5) ? 0 : 1;

        std::cout << "Guessed Results: " << result_vals[0] << std::endl;
        std::cout << "Error: " << my_net.GetError() << std::endl << std::endl;
    }
}

void _AimpliesB_And_Not_CimpliesD_()  /// (A=>B) && !(C=>D)
{
    //creating the topology for the network
    std::vector<size_t> topology;
    topology = { 4,4,2,1 };

    NeuralNet my_net(topology);

    //defining variables
    unsigned num_of_tests = 10000;
    std::vector<double> input_vals(topology.front());
    std::vector<double> target_vals(topology.back());
    std::vector<double> result_vals(topology.back());
    unsigned correct = 0;
    unsigned wrong = 0;

    //doing the training
    for (int i = 1; i <= num_of_tests; ++i) {

        //input the testing input-data
        for (auto& value : input_vals) {
            std::cin >> value;
        }

        //input the testing target-data
        for (auto& value : target_vals) {
            std::cin >> value;
        }

        //output what the inputs were for the given test
        std::cout << "Test #" << i << " Inputs: ";

        //output the inputted data
        for (auto& value : input_vals) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        //output the target data
        std::cout << "Target Value: ";
        for (auto& value : target_vals) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        my_net.FeedForward(input_vals);
        my_net.BackPropagate(target_vals);
        result_vals = my_net.GetResults();

        result_vals[0] = (result_vals[0] < 0.5) ? 0 : 1;

        std::cout << "Guessed Results: " << result_vals[0] << std::endl;
        std::cout << "Error: " << my_net.GetError() << std::endl << std::endl;

        if (result_vals[0] == target_vals[0]) {
            ++correct;
        }
        else {
            ++wrong;
        }
    }

    std::cout << "correct: " << correct << " wrong: " << wrong << " %correct: " << double(correct) / (correct + wrong) * 100 << std::endl;
}

void OutputBarNetwork()
{
    //4 3 3 2
    NeuralNet bar({3, 2, 2, 1});
    bar.FeedForward({ 1.0, 1.0, 0.0 });
    bar.ExportNetworkFile("bar_data.txt");
}

void InputBarNetwork()
{
    NeuralNet bar({0, 0, 0, 0});
    bar.ImportNetworkFile("bar_data.txt");
    bar.PrintLayerInfo();
}

int main() 
{
    //xorTest();
    //_AimpliesB_And_Not_CimpliesD_();
    //OutputBarNetwork();
    //InputBarNetwork();
    PresetNetworkTests::LogicNetworkTest();
}