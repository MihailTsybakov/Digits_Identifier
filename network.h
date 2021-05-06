#ifndef DIDENT
#define DIDENT

#include <vector>
#include <cmath>
#include <string>
#include <ctime>
#include <sstream>
#include <fstream>
#include <iostream>
#include "aux_funcs.h"
#include "matrix.h"
#include "digit_container.h"
#include "layer.h"

class perceptron
{
private:
    double val;
public:
    perceptron() { this->val = 0.0; }
    void setval(double val) { this->val = val; }
    double getval() { return val; }
};

class network
{
private:
    std::vector<network_layer> layers;
    std::vector<perceptron> input_layer;

    std::pair<std::vector<double>, std::vector<double>> feedforward_step(int l, std::vector<double> previous_activations);
    std::vector<double> backpropagation_step(int l, std::vector<double> weighted_sum, std::vector<double> next_delta);
    std::vector<double> lastlayer_delta(std::vector<double> true_ans, std::vector<double> activations,
                        std::vector<double> weighted_sum);

    void SGD_step(std::vector<double> true_ans,  // Stohastic Gradiend Descent learning step
         std::vector<double> perception_input,
         double learning_rate, int batch_size);

    std::function<double(double)> activation = sigmoid;
    std::function<double(double)> d_activation = d_sigmoid;
    std::vector<double> sgd_vector(int digit);
public:
    network(int perception_neurons, std::vector<int> hidden_layers, int output_neurons);
    void store(std::string path, std::string filename_template) const;
    void load(std::string path, std::string filename_template, int layer_num);

    std::vector<double> form_perception(digit_container DC);
    std::vector<double> raw_identify(digit_container DC); // Returns Probability vector
    int identify(digit_container DC); // Returns identified digit
    double batch_test(std::vector<digit_container> batch, std::vector<int> true_answers);
    
    void SGD_learn(std::vector<digit_container> training_samples, std::vector<int> true_answers, double learning_rate,
         int epoch_count, int batch_size, bool epoch_logs, bool dynamic_gamma, double gamma_factor, int update_frequency,
         std::vector<digit_container> test_batch, std::vector<int> test_answers);
};

#endif // DIDENT
