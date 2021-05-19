#ifndef MATRIX
#define MATRIX

#include <vector>
#include <iostream>

#include "aux_funcs.h"

class matrix
{
public:
	size_t row_dim, col_dim;
	std::vector<std::vector<double>> M;

	matrix();
	matrix(size_t rows, size_t cols);
	void random_fill(double mean, double sigma);
	void resize(size_t new_rows, size_t new_cols);
	void print() const;
	matrix transpose() const;
	std::vector<double> mult_by_v(std::vector<double> v) const;
};

#endif//MATRIX
