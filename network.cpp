#include "network.h"

///* ========================================================================================================== */
///* ========================================================================================================== */
///* ========================================================================================================== */

network::network(int perception_neurons, std::vector<int> hidden_layers, int output_neurons)
{
    // hidden_layers.size() = number of hidden layers, each vector element = size of hidden layer
    input_layer.resize(perception_neurons);
    int tmp_layer_dim = perception_neurons;
    for (size_t i = 0; i < hidden_layers.size(); ++i)
    {
        layers.push_back(network_layer(hidden_layers[i], tmp_layer_dim));
        tmp_layer_dim = hidden_layers[i];
    }
    layers.push_back(network_layer(output_neurons, tmp_layer_dim));
}

void network::store(std::string path, std::string filename_template) const
{
    for (size_t i = 0; i < layers.size(); ++i)
    {
        std::stringstream ss;
        ss << path << "\\" << filename_template << "_" << i << ".txt";
        layers[i].store_layer(ss.str());
    }
    std::cout << "<logs> Network stored." << std::endl;
}

void network::load(std::string path, std::string filename_template, int layer_num)
{
    for (int i = 0; i < layer_num; ++i)
    {
        std::stringstream ss;
        ss << path << "\\" << filename_template << "_" << i << ".txt";
        layers[i].load_layer(ss.str());
    }
    std::cout << "<logs> Network loaded." << std::endl;
}

std::pair<std::vector<double>, std::vector<double>> network::feedforward_step(int l, std::vector<double> previous_activations)
{
    // Returns current weighted sum (pair.first) and current neuron activations (pair.second)
    std::vector<double> curr_WS = layers[l].W.mv_mult(previous_activations) + layers[l].bias; // Weighted sums
    std::vector<double> current_activations = apply_f(curr_WS, activation); // Current neuron layer activations
    return std::pair<std::vector<double>, std::vector<double>>(curr_WS, current_activations);
}

std::vector<double> network::backpropagation_step(int l, std::vector<double> weighted_sum, std::vector<double> next_delta)
{
    // Calculates layer l's error (= delta) with weighted sum and delta of layer l+1
    std::vector<double> dsigma = apply_f(weighted_sum, d_activation);  // d_
    matrix transposed_weights = (layers[l + 1].W).transpose();
    std::vector<double> matrix_product = transposed_weights.mv_mult(next_delta);
    std::vector<double> delta_l = hadamard_product(matrix_product, dsigma);
    return delta_l;
}

std::vector<double> network::lastlayer_delta(std::vector<double> true_ans, std::vector<double> activations, std::vector<double> weighted_sum)
{
    std::vector<double> ll_delta = hadamard_product((activations - true_ans), apply_f(weighted_sum, d_activation)); // d_
    return ll_delta;
}

std::vector<double> network::form_perception(digit_container DC)
{
    std::vector<double> perc;
    int h = DC.dim().first;
    int w = DC.dim().second;
    perc.resize(h * w);
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            perc[i * w + j] = DC.get(j, i);
        }
    }
    return perc;
}

std::vector<double> network::raw_identify(digit_container DC) // Returns Probability vector
{
    // Reading data from digit container
    std::vector<double> input_perception = form_perception(DC);
    std::vector<double> tmp_activations;
    tmp_activations = input_perception;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        std::pair<std::vector<double>, std::vector<double>> fws_result = feedforward_step(i, tmp_activations);
        tmp_activations = fws_result.second;
    }
    return tmp_activations;
}

int network::identify(digit_container DC) // Returns identified digit
{
    std::vector<double> digits_probabilities = raw_identify(DC);
    int choosen_digit = 0;
    for (int i = 0; i < 10; ++i)
    {
        if (digits_probabilities[i] > digits_probabilities[choosen_digit])
        {
            choosen_digit = i;
        }
    }
    return choosen_digit;
}

double network::batch_test(std::vector<digit_container> batch, std::vector<int> true_answers)
{
    int count = 0;
    for (size_t i = 0; i < true_answers.size(); ++i)
    {
        if (identify(batch[i]) == true_answers[i]) count++;
    }
    return static_cast<double>(count) / true_answers.size();
}

void network::SGD_learn(std::vector<digit_container> trainig_samples, std::vector<int> true_answers, double learning_rate,
     int epoch_count, int batch_size, bool epoch_logs, bool dynamic_gamma, double gamma_factor, int update_frequency,
     std::vector<digit_container> test_batch, std::vector<int> test_answers)
{
    if (true_answers.size() != trainig_samples.size())
    {
        std::cout << "Error: dimension mismatch in SGD." << std::endl;
        exit(-1);
    }
    srand(time(nullptr));
    int gamma_upd = 0;
    for (int epoch_num = 0; epoch_num < epoch_count; ++epoch_num)
    {
        gamma_upd++;
        // Logging
        if (epoch_logs == true)
        {
            std::cout << "<logs> Before epoch " << epoch_num + 1 << ": accuracy is " << batch_test(test_batch, test_answers) << std::endl;
            std::cout << "<logs> Learning epoch " << epoch_num + 1 << " started." << std::endl;
            std::cout << "<logs> Learning rate during current epoch: " << learning_rate << std::endl;
        }
        // Updating learning rate
        if (dynamic_gamma == true && gamma_upd >= update_frequency)
        {
            learning_rate *= gamma_factor;
            gamma_upd = 0;
        }
        // Forming mini-batches
        for (int batch_num = 0; batch_num < trainig_samples.size() / batch_size; ++batch_num)
        {
            for (int sample_num = 0; sample_num < batch_size; ++sample_num)
            {
                int index = rand() % trainig_samples.size();
                digit_container train_dc = trainig_samples[index];
                std::vector<double> sgd_vect = sgd_vector(true_answers[index]);
                std::vector<double> sgd_perc = form_perception(train_dc);
                SGD_step(sgd_vect, sgd_perc, learning_rate, batch_size);
            }
        }
    }
}

void network::SGD_step(std::vector<double> true_ans,  // Stohastic Gradiend Descent learning step
    std::vector<double> perception_input,
    double learning_rate, int batch_size)
{
    std::vector<std::vector<double>> layer_activations;     // al
    std::vector<std::vector<double>> layer_weighted_sums;   // zl
    std::vector<std::vector<double>> layer_deltas;          // dl
    std::vector<double> tmp_activations = perception_input;
    // feeding perception forward through net
    for (size_t i = 0; i < layers.size(); ++i)
    {
        std::pair<std::vector<double>, std::vector<double>> fws_result = feedforward_step(i, tmp_activations);
        layer_weighted_sums.push_back(fws_result.first);
        layer_activations.push_back(fws_result.second);
        tmp_activations = fws_result.second;
    }
    std::vector<double> ll_delta = lastlayer_delta(true_ans, tmp_activations, layer_weighted_sums[layers.size() - 1]);
    layer_deltas.resize(layers.size());
    layer_deltas[layers.size() - 1] = ll_delta; // Definig last layer's delta
    for (int i = static_cast<int>(layers.size()) - 2; i >= 0; --i)
    {
        layer_deltas[i] = backpropagation_step(i, layer_weighted_sums[i], layer_deltas[i + 1]);
    }
    // Updating weights and biases:
    for (size_t i = 0; i < layers.size(); ++i)
    {
        // Bias update:
        for (size_t j = 0; j < layers[i].bias.size(); ++j) layers[i].bias[j] -= learning_rate * layer_deltas[i][j] / batch_size;
        // Weights update:
        std::vector<std::vector<double>> old_weights = layers[i].W.get_m();
        std::vector<double> activations;
        if (i == 0)
        {
            activations = perception_input;
        }
        else
        {
            activations = layer_activations[i - 1];
        }
        for (int p_ = 0; p_ < layers[i].W.dim().first; ++p_)
        {
            for (int q_ = 0; q_ < layers[i].W.dim().second; ++q_)
            {
                layers[i].W.set(p_, q_, old_weights[p_][q_] - learning_rate * activations[q_] * layer_deltas[i][p_]/batch_size);
            }
        }
    }
}

std::vector<double> network::sgd_vector(int digit)
{
    std::vector<double> res; res.resize(10);
    for (int i = 0; i < 10; ++i) res[i] = 0.0;
    res[digit] = 1.0;
    return res;
}
