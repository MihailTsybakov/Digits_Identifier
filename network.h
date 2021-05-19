#ifndef NETWORK
#define NETWORK

#include <sstream>
#include <thread>
#include <mutex>
#include <ctime>

#include "layer.h"
#include "digit_container.h"

typedef struct
{
public:
	int thread_number;
	int epoch_count;
	int batch_size;
	int update_frequency;
	double learning_rate;
	double mult_factor;
	double L2_lambda;
	bool epoch_logs;
	bool dynamic_LR;
	std::vector<digit_container> train_images;
	std::vector<digit_container> test_images;
	std::vector<int> train_answers;
	std::vector<int> test_answers;
	std::mutex* mtx;

}   learn_instructor;

typedef struct
{
public:
	int batch_size;
	double learning_rate;
	double L2_lambda;
	std::vector<double> true_ans;
	std::vector<double> perception;
	std::mutex* mtx;
	
}	sgd_instructor;

typedef struct
{
public:
	std::function<double(double)> activation;
	std::function<double(double)> d_activation;
	int perception_neurons;
	int output_neurons;
	std::vector<int> hidden_layers;

}   create_instructor;

class network
{
private:
	std::vector<layer> layers;
	std::function<double(double)> activation;
	std::function<double(double)> d_activation;

	std::vector<double> feedforward(std::vector<double> perception);
	void sgd_step(sgd_instructor si);
public:
	network(create_instructor ci);
	void store(std::string network_name) const;
	void load(std::string network_name, int layer_count);
	void sgd_learn(learn_instructor li);
	void sgd_parallel_learn(learn_instructor li);
	double batch_test(std::vector<digit_container> imgs, std::vector<int> labels);
	int identify(digit_container dc);
};


#endif//NETWORK
