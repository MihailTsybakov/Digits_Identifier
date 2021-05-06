#ifndef AUXF
#define AUXF

#include <vector>
#include <cmath>
#include <string>
#include <ctime>
#include <sstream>
#include <fstream>
#include <iostream>
#include <random>
#include <functional>

double random_number(double mean, double standard_deviation);

double MSE(std::vector<double> y_true, std::vector<double> y_pred); // Absolute squared error

double accuracy(std::vector<int> y_true, std::vector<int> y_pred);

double scalmul(std::vector<double> v1, std::vector<double> v2); // Scalar Multiplication

std::vector<double> random_vector(int size);

std::vector<double> hadamard_product(std::vector<double> v1, std::vector<double> v2);

std::vector<double> operator+(std::vector<double> v1, std::vector<double> v2);

std::vector<double> operator-(std::vector<double> v1, std::vector<double> v2);

std::vector<double> apply_f(std::vector<double> v, std::function<double(double)> func); // Applies function to a vector

///*  ============= Activations ============ */

double sigmoid(double x);

double d_sigmoid(double x); // Sigmoid derivative

#endif // AUXF
