#include "aux_funcs.h"

double random_number(double mean, double standard_deviation)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> nd(mean, standard_deviation); ///* Median & Standard Deviation */
    return nd(gen);
}

double MSE(std::vector<double> y_true, std::vector<double> y_pred) // Absolute squared error
{
    double R = 0;
    for (size_t i = 0; i < y_true.size(); ++i) R += pow((y_true[i] - y_pred[i]), 2);
    return R / static_cast<double>(y_true.size());
}

double accuracy(std::vector<int> y_true, std::vector<int> y_pred)
{
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        if (y_true[i] == y_pred[i]) correct++;
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

double scalmul(std::vector<double> v1, std::vector<double> v2) // Scalar Multiplication
{
    if (v1.size() != v2.size())
    {
        std::cout << "Error: wrong vectors size." << std::endl;
        exit(-1);
    }
    double r = 0;
    for (size_t i = 0; i < v1.size(); ++i) r += v1[i] * v2[i];
    return r;
}

std::vector<double> random_vector(int size)
{
    std::vector<double> res; res.resize(size);
    for (int i = 0; i < size; ++i) res[i] = random_number(0, 1);
    return res;
}

std::vector<double> hadamard_product(std::vector<double> v1, std::vector<double> v2)
{
    if (v1.size() != v2.size())
    {
        std::cout << "Error: wrong dimensions in Hadamard product." << std::endl;
        exit(-1);
    }
    std::vector<double> res; res.resize(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) res[i] = v1[i] * v2[i];
    return res;
}

std::vector<double> operator+(std::vector<double> v1, std::vector<double> v2)
{
    if (v1.size() != v2.size())
    {
        std::cout << "Error: wrong dimensions for operator+." << std::endl;
        exit(-1);
    }
    std::vector<double> res; res.resize(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) res[i] = v1[i] + v2[i];
    return res;
}

std::vector<double> operator-(std::vector<double> v1, std::vector<double> v2)
{
    if (v1.size() != v2.size())
    {
        std::cout << "Error: wrong dimensions for operator-." << std::endl;
        exit(-1);
    }
    std::vector<double> res; res.resize(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) res[i] = v1[i] - v2[i];
    return res;
}


/* ======================================================================= */
/* ======================================================================= */

std::vector<double> apply_f(std::vector<double> v, std::function<double(double)> func) // Applies function to a vector
{
    std::vector<double> res; res.resize(v.size());
    for (size_t i = 0; i < v.size(); ++i) res[i] = func(v[i]);
    return res;
}

///*  ============= Activations ============ */

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double d_sigmoid(double x) // Sigmoid derivative
{
    return exp(-x) / pow((1 + exp(-x)), 2);
}