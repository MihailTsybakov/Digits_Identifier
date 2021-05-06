#ifndef MNIST_DECODER
#define MNIST_DECODER

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "digit_container.h"

std::vector<digit_container> MNIST_images(std::string file_path, std::string filename, int sample_num)
{
	std::vector<digit_container> res;
	std::ifstream mnist_dataset;
	mnist_dataset.open(file_path + "\\" + filename, ios::in | ios::binary);
	if (!mnist_dataset.is_open())
	{
		std::cout << "Error: cannot open " << file_path + "\\" + filename << std::endl;
		exit(-1);
	}

	uint8_t pixel;
	for (int i = 0; i < sample_num; ++i)
	{
		mnist_dataset.seekg(16 + 784*i);
		digit_container tmp_DC(28, 28);
		for (int j = 0; j < 28 * 28; ++j)
		{
			mnist_dataset >> pixel;
			tmp_DC.set(j%28, j/28, pixel);
		}
		res.push_back(tmp_DC);
	}

	mnist_dataset.close();
	return res;
}

std::vector<int> MNIST_labels(std::string file_path, std::string filename, int sample_num)
{
	std::vector<int> res;
	std::ifstream mnist_labels;
	mnist_labels.open(file_path + "\\" + filename, ios::in);
	if (!mnist_labels.is_open())
	{
		std::cout << "Error: cannot open " << file_path + "\\" + filename << std::endl;
		exit(-1);
	}

	int label;
	for (int i = 0; i < sample_num; ++i)
	{
		mnist_labels >> label;
		res.push_back(static_cast<int>(label));
	}

	mnist_labels.close();
	return res;
}

#endif//MNIST_DECODER
