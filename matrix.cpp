#include "matrix.h"

void matrix::print() const
{
    for (int i = 0; i < dim_h; ++i)
    {
        for (int j = 0; j < dim_w; ++j) std::cout << M[i][j] << " ";
        std::cout << std::endl;
    }
}

matrix matrix::transpose() const
{
    matrix res(dim_w, dim_h);
    for (int i = 0; i < dim_h; ++i)
    {
        for (int j = 0; j < dim_w; ++j) res.set(j, i, M[i][j]);
    }
    return res;
}

std::pair<int, int> matrix::dim() const
{
    return std::pair<int, int>(dim_h, dim_w);
}

matrix::matrix(int h, int w)
{
    M.resize(h);
    for (int i = 0; i < h; ++i) M[i].resize(w);
    dim_h = h;
    dim_w = w;
}

void matrix::random_fill()
{
    for (int i = 0; i < dim_h; ++i)
    {
        for (int j = 0; j < dim_w; ++j)
        {
            M[i][j] = random_number(0, 1/sqrt(dim_w));
        }
    }
}

void matrix::resize(int h, int w)
{
    M.resize(h);
    dim_h = h;
    for (int i = 0; i < h; ++i) M[i].resize(w);
    dim_w = w;
}

void matrix::set(int i, int j, double val)
{
    M[i][j] = val;
}

void matrix::set_m(std::vector<std::vector<double>> M)
{
    dim_h = static_cast<int>(M.size());
    dim_w = static_cast<int>(M[0].size());
    this->M = M;
}

std::vector<std::vector<double>> matrix::get_m() const
{
    return M;
}

std::vector<double> matrix::mv_mult(std::vector<double> v)
{
    if (dim_w != v.size())
    {
        std::cout << "Error: wrong dimensions in mv_mult." << std::endl;
        exit(-1);
    }
    std::vector<double> res;
    for (int i = 0; i < dim_h; ++i)
    {
        res.push_back(scalmul(v, M[i]));
    }
    return res;
}