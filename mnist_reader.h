#ifndef MNIST_READER
#define MNIST_READER

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "digit_container.h"

class mnist_reader
{
private:
	std::string path;
	std::string labels_file;
	std::string images_file;
public:
	mnist_reader(std::string path, std::string labels_file, std::string images_file);
	std::vector<digit_container> read_images(int img_number);
	std::vector<int> read_labels(int lbl_number);
};

#endif//MNIST_READER
