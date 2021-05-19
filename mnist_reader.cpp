#include "mnist_reader.h"

mnist_reader::mnist_reader(std::string path, std::string labels_file, std::string images_file)
{
	this->path = path;
	this->labels_file = labels_file;
	this->images_file = images_file;
}

std::vector<digit_container> mnist_reader::read_images(int img_number)
{
	std::vector<digit_container> res;
	std::ifstream mnist_dataset;
	mnist_dataset.open(path + "\\" + images_file, std::ios::binary);
	if (!mnist_dataset.is_open())
	{
		std::cout << "Error: cannot open images file." << std::endl;
		exit(-1);
	}

	char pixel;
	mnist_dataset.seekg(15);
	for (int i = 0; i < img_number; ++i)
	{
		//mnist_dataset.seekg(16 + 784*i);
		digit_container tmp_DC(28, 28);
		for (int j = 0; j < 28 * 28; ++j)
		{
			mnist_dataset.read(&pixel, sizeof(char));
			tmp_DC.set(j % 28, j / 28, pixel);
		}
		res.push_back(tmp_DC);
	}

	mnist_dataset.close();
	return res;
}

std::vector<int> mnist_reader::read_labels(int lbl_number)
{
	std::vector<int> res;
	std::ifstream mnist_labels;
	mnist_labels.open(path + "\\" + labels_file, ios::in);
	if (!mnist_labels.is_open())
	{
		std::cout << "Error: cannot open labels file." << std::endl;
		exit(-1);
	}

	int label;
	for (int i = 0; i < lbl_number; ++i)
	{
		mnist_labels >> label;
		res.push_back(static_cast<int>(label));
	}

	mnist_labels.close();
	return res;
}