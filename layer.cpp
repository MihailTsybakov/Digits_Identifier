#include "layer.h"

layer::layer(size_t layer_size, size_t prev_size)
{
	this->layer_size = layer_size;
	this->prev_size = prev_size;
	W.resize(layer_size, prev_size);
	W.random_fill(static_cast<double>(MEAN), static_cast<double>(SIGMA));
	bias = random_vector(static_cast<double>(MEAN), static_cast<double>(SIGMA), layer_size);
}

void layer::save(std::string filename) const
{
	if (filename.find(".txt") == std::string::npos) std::cout << "Error: unsupported file format." << std::endl, exit(-1);
	std::ofstream out_file; out_file.open(filename);
	if (!out_file.is_open()) std::cout << "Error: cannot open " << filename << std::endl, exit(-1);
	out_file << layer_size << " " << prev_size << std::endl;
	for (size_t row = 0; row < layer_size; ++row)
	{
		for (size_t col = 0; col < prev_size; ++col) out_file << W.M[row][col] << " ";
		out_file << bias[row] << std::endl;
	}
	out_file.close();
	std::cout << "<logs> Layer stored." << std::endl;
}

void layer::load(std::string filename)
{
	if (filename.find(".txt") == std::string::npos) std::cout << "Error: unsupported file format." << std::endl, exit(-1);
	std::ifstream in_file; in_file.open(filename);
	if (!in_file.is_open()) std::cout << "Error: cannot open " << filename << std::endl, exit(-1);
	size_t l_size, p_size;
	in_file >> l_size; in_file >> p_size;
	W.resize(l_size, p_size);
	for (size_t row = 0; row < l_size; ++row)
	{
		for (size_t col = 0; col < p_size; ++col) in_file >> W.M[row][col];
		in_file >> bias[row];
	}
	in_file.close();
	std::cout << "<logs> Layer loaded." << std::endl;
}
