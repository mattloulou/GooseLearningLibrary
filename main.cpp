#include "neural_net.h"
#include <iostream>

using namespace std;

void xorTest()
{
    //creating the topology for the network
    std::vector<size_t> topology;
    topology = { 2,2,1 };

    NeuralNet my_net(topology);

    //defining variables
    unsigned numOfTests = 10000;
    std::vector<double> input_vals = { 0, 0 };
    std::vector<double> target_vals = { 0 };
    std::vector<double> result_vals = { 0 };

    //doing the training
    for (int i = 1; i <= numOfTests; ++i) {
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


int main() 
{
    xorTest();
}