#ifndef LAYER
#define LAYER

#include "matrix.h"
#include "aux_funcs.h"
//#include "network.h"

class network;

class network_layer
{
private:
	matrix W;
	std::vector<double> bias;
public:
	network_layer(int dim_h, int dim_w);
	void store_layer(std::string filename) const;
	void load_layer(std::string filename);



	size_t dim() const noexcept;
	friend class network;

};

#endif //LAYER
