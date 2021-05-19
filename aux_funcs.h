#ifndef AUXF
#define AUXF

#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include <cmath>

#include "digit_container.h"

#define MEAN 0
#define SIGMA 1

double scalar_product(std::vector<double> v1, std::vector<double> v2);

std::vector<double> hadamard_product(std::vector<double> v1, std::vector<double> v2);

double random_number(double mean, double sigma);

std::vector<double> random_vector(double mean, double sigma, int size);

std::vector<double> apply(std::vector<double> v, std::function<double(double)> f);

std::vector<double> form_perception(digit_container dc);

void print_v(std::vector<double> v);

std::vector<double> operator+(std::vector<double> v1, std::vector<double> v2);
std::vector<double> operator-(std::vector<double> v1, std::vector<double> v2);
std::vector<double> operator*(double m, std::vector<double> v);

double sigmoid(double val);
double d_sigmoid(double val);

#endif//AUXF
