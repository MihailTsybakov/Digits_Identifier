#include "layer.h"

size_t network_layer::dim() const noexcept
{
    return W.dim().first;
}

network_layer::network_layer(int dim_h, int dim_w) // Random initialization
{
    W.resize(dim_h, dim_w);
    bias = random_vector(dim_h);
    W.random_fill();
}

void network_layer::store_layer(std::string filename) const // Saving layer's weights and biases
{
    std::ofstream output_file;
    output_file.open(filename, std::ios::out);
    if (!output_file.is_open())
    {
        std::cout << "Error: cannot create " << filename << std::endl;
        exit(-1);
    }
    output_file << W.dim().first << " " << W.dim().second << std::endl; // Writing layer shape
    std::vector<std::vector<double>> W_ = W.get_m();
    for (size_t i = 0; i < W.dim().first; ++i)
    {
        for (size_t j = 0; j < W.dim().second; ++j)
        {
            output_file << W_[i][j] << " ";
        }
        output_file << bias[i] << std::endl;
    }
    std::cout << "<logs> Layer " << filename << " stored." << std::endl;
    output_file.close();
}

void network_layer::load_layer(std::string filename) // Loading layer's weights and biases
{
    std::ifstream input_file;
    input_file.open(filename, std::ios::in);
    if (!input_file.is_open())
    {
        std::cout << "Error: cannot open " << filename << std::endl;
        exit(-1);
    }
    int layer_dim; input_file >> layer_dim;
    int prev_dim; input_file >> prev_dim;
    W.resize(layer_dim, prev_dim);
    bias.resize(layer_dim);
    for (int i = 0; i < layer_dim; ++i)
    {
        for (int j = 0; j < prev_dim; ++j)
        {
            double tmp;
            input_file >> tmp;
            W.set(i, j, tmp);
        }
        input_file >> bias[i];
    }
    std::cout << "<logs> Layer " << filename << " loaded." << std::endl;
    input_file.close();
}
