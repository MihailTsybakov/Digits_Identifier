#include "matrix.h"

matrix::matrix(){}

matrix::matrix(size_t rows, size_t cols)
{
	M.resize(rows); 
	row_dim = rows;
	col_dim = cols;
	for (auto row = M.begin(); row != M.end(); ++row) row->resize(cols);
}

void matrix::random_fill(double mean, double sigma)
{
	for (size_t row = 0; row < M.size(); ++row)
	{
		for (size_t col = 0; col < M[0].size(); ++col) M[row][col] = random_number(0, 1 / sqrt(col_dim));
	}
}

matrix matrix::transpose() const
{
	matrix r_matrix(col_dim, row_dim);
	for (size_t row = 0; row < row_dim; ++row)
	{
		for (size_t col = 0; col < col_dim; ++col) r_matrix.M[col][row] = M[row][col];
	}
	return r_matrix;
}

std::vector<double> matrix::mult_by_v(std::vector<double> v) const
{
	if (v.size() != col_dim) std::cout << "Error: dimension mismatch in mult by vector." << std::endl, exit(-1);
	std::vector<double> r_vect;
	for (auto row : M) r_vect.push_back(scalar_product(row, v));
	return r_vect;
}

void matrix::resize(size_t new_rows, size_t new_cols)
{
	row_dim = new_rows;
	col_dim = new_cols;
	M.resize(new_rows);
	for (auto row = M.begin(); row != M.end(); ++row) row->resize(new_cols);
}

void matrix::print() const
{
	for (auto row : M)
	{
		for (auto element : row) std::cout << element << " ";
		std::cout << std::endl;
	}
}
