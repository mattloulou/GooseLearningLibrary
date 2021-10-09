#include "neural_net.h"
#include "preset_network_test.h"
#include <iostream>

using namespace std;

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
    //OutputBarNetwork();
    //InputBarNetwork();
    PresetNetworkTests::LogicNetworkTest();
}