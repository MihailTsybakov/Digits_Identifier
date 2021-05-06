#ifndef MATRIX
#define MATRIX

#include <vector>
#include <cmath>
#include <string>
#include <ctime>
#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include "aux_funcs.h"

class matrix
{
private:
    int dim_h, dim_w;
    std::vector<std::vector<double>> M;
public:
    matrix() {}
    matrix(int h, int w);
    
    void random_fill();
    void resize(int h, int w);
    void set(int i, int j, double val);
    void set_m(std::vector<std::vector<double>> M);
    std::vector<std::vector<double>> get_m() const;
    std::vector<double> mv_mult(std::vector<double> v);
    std::pair<int, int> dim() const;
    matrix transpose() const;
    void print() const;
    
};

#endif //MATRIX
