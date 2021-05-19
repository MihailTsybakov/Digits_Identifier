#include "aux_funcs.h"

double random_number(double mean, double sigma)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> norm_distr(mean, sigma);
	return norm_distr(gen);
}

double scalar_product(std::vector<double> v1, std::vector<double> v2)
{
	if (v1.size() != v2.size()) std::cout << "Error: dimension mismatch in scalar prod." << std::endl, exit(-1);
	double sum = 0;
	for (size_t i = 0; i < v1.size(); ++i) sum += v1[i] * v2[i];
	return sum;
}

std::vector<double> hadamard_product(std::vector<double> v1, std::vector<double> v2)
{
	if (v1.size() != v2.size()) std::cout << "Error: dimension mismatch in hadamard prod." << std::endl, exit(-1);
	std::vector<double> h_prod; h_prod.resize(v1.size());
	for (size_t i = 0; i < v1.size(); ++i) h_prod[i] = v1[i] * v2[i];
	return h_prod;
}

std::vector<double> random_vector(double mean, double sigma, int size)
{
	std::vector<double> r_vect; r_vect.resize(size);
	for (int i = 0; i < size; ++i) r_vect[i] = random_number(mean, sigma);
	return r_vect;
}

std::vector<double> apply(std::vector<double> v, std::function<double(double)> f)
{
	std::vector<double> res;
	for (auto val : v) res.push_back(f(val));
	return res;
}

std::vector<double> form_perception(digit_container dc)
{
	std::vector<double> perc;
	int h = dc.dim().first;
	int w = dc.dim().second;
	perc.resize(h * w);
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			perc[i * w + j] = dc.get(j, i);
		}
	}
	return perc;
}

void print_v(std::vector<double> v)
{
	std::cout << "[";
	for (auto el : v) std::cout << el << " ";
	std::cout << "]" << std::endl;
}

std::vector<double> operator+(std::vector<double> v1, std::vector<double> v2)
{
	if (v1.size() != v2.size()) std::cout << "Error: dimension mismatch in operator+." << std::endl, exit(-1);
	std::vector<double> sum_v; sum_v.resize(v1.size());
	for (size_t i = 0; i < v1.size(); ++i) sum_v[i] = v1[i] + v2[i];
	return sum_v;
}

std::vector<double> operator-(std::vector<double> v1, std::vector<double> v2)
{
	if (v1.size() != v2.size()) std::cout << "Error: dimension mismatch in operator-." << std::endl, exit(-1);
	std::vector<double> sum_v; sum_v.resize(v1.size());
	for (size_t i = 0; i < v1.size(); ++i) sum_v[i] = v1[i] - v2[i];
	return sum_v;
}

std::vector<double> operator*(double m, std::vector<double> v)
{
	std::vector<double> mult_v; mult_v.resize(v.size());
	for (size_t i = 0; i < v.size(); ++i) mult_v[i] = v[i]*m;
	return mult_v;
}

double sigmoid(double val)
{
	return (1 / (1 + exp(-val)));
}

double d_sigmoid(double val)
{
	return sigmoid(val) * (1 - sigmoid(val));
}
