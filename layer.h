#ifndef LAYER
#define LAYER

#include <fstream>
#include <string>

#include "matrix.h"

class network;
class layer
{
private:
	size_t layer_size;
	size_t prev_size;
	std::vector<double> bias;
	matrix W;
public:
	layer(){}
	layer(size_t layer_size, size_t prev_size);
	void save(std::string filename) const;
	void load(std::string filename);
	friend class network;
};

#endif//LAYER

