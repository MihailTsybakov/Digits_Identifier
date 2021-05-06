#include <iostream>
#include <random>
#include "digit_container.h"
#include "network.h"
#include "MNIST_decoder.h"


int main(int argc, char* argv[])
{
    // Initializaing neural net
    std::vector<int> hidden_layer_params = { 100 };
    network N(28 * 28, hidden_layer_params, 10);

    // Getting training & testing data
    std::vector<digit_container> train_images = MNIST_images("C:\\Users\\mihai\\Desktop\\progy\\C & C++\\Digits_Identifier\\Data",
        "train-images.idx3-ubyte", 60'000);
    std::vector<int> train_labels = MNIST_labels("C:\\Users\\mihai\\Desktop\\progy\\C & C++\\Digits_Identifier\\Data",
        "train_labels.txt", 60'000);
    std::vector<digit_container> test_images = MNIST_images("C:\\Users\\mihai\\Desktop\\progy\\C & C++\\Digits_Identifier\\Data",
        "t10k-images.idx3-ubyte", 1000);
    std::vector<int> test_labels = MNIST_labels("C:\\Users\\mihai\\Desktop\\progy\\C & C++\\Digits_Identifier\\Data",
        "test_labels.txt", 1000);

    //// Trainig neural network
    //N.SGD_learn(train_images, train_labels, 3, 24, 10, true, true, 0.8, 4, test_images, test_labels);

    //// Saving weights
    //N.store("C:\\Users\\mihai\\Desktop\\progy\\Additional\\Neural_Networks_Dump\\Digits_Identifier", "Net_Dump_Big");

    N.load("C:\\Users\\mihai\\Desktop\\progy\\Additional\\Neural_Networks_Dump\\Digits_Identifier",
        "0_957", 2);
    


    // Precision test
    std::vector<int> network_predictions;
    for (size_t i = 0; i < test_labels.size(); ++i)
    {
        network_predictions.push_back(N.identify(test_images[i]));
    }
    int correct_answers = 0;
    for (size_t i = 0; i < test_labels.size(); ++i)
    {
        if (network_predictions[i] == test_labels[i])
        {
            correct_answers++;
        }
    }
    std::cout << "Accuracy: " << static_cast<double>(correct_answers) / test_labels.size() << std::endl;

    std::cout << "****************" << std::endl;

    for (int i = 0; i < 20; ++i)
    {
        std::stringstream ss;
        ss << "Test_" << i + 1 << ".bmp";
        test_images[i].save(ss.str());
        std::cout << "Sample " << i << ": predict = " << N.identify(test_images[i]) << " (" << test_labels[i] << ")" << std::endl;
    }

    return 0;
}