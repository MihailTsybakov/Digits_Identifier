#include "network.h"

network::network(create_instructor ci)
{
	this->activation = ci.activation;
	this->d_activation = ci.d_activation;
	int prevlayer_size = ci.perception_neurons;
	for (auto layer_size : ci.hidden_layers)
	{
		layers.push_back(layer(layer_size, prevlayer_size));
		prevlayer_size = layer_size;
	}
	layers.push_back(layer(ci.output_neurons, prevlayer_size));
}

void network::store(std::string network_name) const
{
	for (size_t i = 0; i < layers.size(); ++i)
	{
		std::stringstream filename;
		filename << network_name << "_" << i << ".txt";
		layers[i].save(filename.str());
	}
	std::cout << "<logs> Network stored." << std::endl;
}

void network::load(std::string network_name, int layer_count)
{
	layers.resize(layer_count);
	for (int i = 0; i < layer_count; ++i)
	{
		std::stringstream filename;
		filename << network_name << "_" << i << ".txt";
		layers[i].load(filename.str());
	}
	std::cout << "<logs> Network loaded." << std::endl;
}


std::vector<double> network::feedforward(std::vector<double> perception)
{
	for (auto layer : layers) perception = apply(layer.W.mult_by_v(perception) + layer.bias, activation);
	return perception;
}


void network::sgd_step(sgd_instructor si)
{
	std::vector<std::vector<double>> neuron_inputs, neuron_activations, layer_deltas;
	std::vector<double> activations = si.perception;
	for (auto layer : layers) // feedforward
	{
		std::vector<double> currlayer_inputs = layer.W.mult_by_v(activations) + layer.bias;
		neuron_inputs.push_back(currlayer_inputs);
		neuron_activations.push_back(apply(currlayer_inputs, activation));
		activations = apply(currlayer_inputs, activation);
	}
	int lnum = static_cast<int>(layers.size());
	layer_deltas.resize(lnum);
	layer_deltas[lnum - 1] = hadamard_product(apply(neuron_inputs[lnum - 1], d_activation), (neuron_activations[lnum - 1] - si.true_ans));
	activations = si.perception;
	for (int i = lnum - 2; i >= 0; --i) // backpropagate
	{
		layer_deltas[i] = hadamard_product(apply(neuron_inputs[i], d_activation), layers[i + 1].W.transpose().mult_by_v(layer_deltas[i + 1]));
	}
	for (int i = 0; i < lnum; ++i) // updating weights and biases
	{
		si.mtx->lock();
		for (size_t j = 0; j < layers[i].layer_size; ++j) layers[i].bias[j] -= (si.learning_rate / si.batch_size) * layer_deltas[i][j];
		for (size_t p = 0; p < layers[i].layer_size; ++p)
		{
			for (size_t q = 0; q < layers[i].prev_size; ++q)
			{
				layers[i].W.M[p][q] -= (si.learning_rate / si.batch_size) * (activations[q] * layer_deltas[i][p]);
			}
		}
		si.mtx->unlock();
		activations = neuron_activations[i]; // ~~
	}
}

void network::sgd_learn(learn_instructor li)
{
	srand(time(nullptr));
	int lr_upd = 0;
	for (int epoch = 0; epoch < li.epoch_count; ++epoch)
	{
		lr_upd++;
		if (li.epoch_logs == true)
		{
			li.mtx->lock();
			std::cout << "<logs> Thread " << std::this_thread::get_id() << ": starting epoch " << epoch + 1 << std::endl;
			std::cout << "<logs> Thread " << std::this_thread::get_id() << ": accuracy is " << batch_test(li.test_images, li.test_answers) << std::endl;
			li.mtx->unlock();
		}
		if (li.dynamic_LR == true && lr_upd >= li.update_frequency)
		{
			lr_upd = 0;
			li.learning_rate *= li.mult_factor;
		}

		for (int batch = 0; batch < li.train_images.size() / li.batch_size; ++batch)
		{
			for (int sample = 0; sample < li.batch_size; ++sample)
			{
				int rand_index = rand() % li.train_answers.size();
				std::vector<double> true_answer; true_answer.resize(10);
				for (int i = 0; i < layers[layers.size() - 1].layer_size; ++i) true_answer[i] = 0.0;
				true_answer[li.train_answers[rand_index]] = 1.0;
				sgd_instructor si; si.batch_size = li.batch_size;
				si.L2_lambda = li.L2_lambda; si.learning_rate = li.learning_rate;
				si.perception = form_perception(li.train_images[rand_index]);
				si.true_ans = true_answer; si.mtx = li.mtx;

				sgd_step(si);
			}
		}
	}
}

void network::sgd_parallel_learn(learn_instructor li)
{
	std::mutex* sgd_mutex = new std::mutex;
	std::vector<std::thread> thread_pool;
	if (li.epoch_count % li.thread_number != 0) std::cout << "Error: epoch number must be completely divided by thread number." << std::endl, exit(-1);
	learn_instructor li_; 
	
	li_.batch_size = li.batch_size; li_.dynamic_LR = li.dynamic_LR;
	li_.epoch_count = li.epoch_count / li.thread_number; li_.epoch_logs = li.epoch_logs;
	li_.L2_lambda = li.L2_lambda; li_.learning_rate = li.learning_rate; li_.mtx = sgd_mutex;
	li_.mult_factor = li.mult_factor; li_.test_answers = li.test_answers; li_.test_images = li.test_images;
	li_.thread_number = 0; li_.train_answers = li.train_answers; li_.train_images = li.train_images;
	li_.update_frequency = li.update_frequency;

	if (li.epoch_logs == true) std::cout << "<logs> Invoking thread pool." << std::endl;
	for (int i = 0; i < li.thread_number; ++i) thread_pool.push_back(std::thread(&network::sgd_learn, this, li_));
	for (auto th = thread_pool.begin(); th != thread_pool.end(); ++th) th->join();
	if (li.epoch_logs == true) std::cout << "<logs> Thread pool joined." << std::endl;

	delete sgd_mutex;
}

int network::identify(digit_container dc)
{
	std::vector<double> conf_vect = feedforward(form_perception(dc));
	int choosen_digit = 0;
	for (int i = 0; i < layers[layers.size() - 1].layer_size; ++i)
	{
		if (conf_vect[i] > conf_vect[choosen_digit]) choosen_digit = i;
	}
	return choosen_digit;
}

double network::batch_test(std::vector<digit_container> imgs, std::vector<int> labels)
{
	if (imgs.size() != labels.size()) std::cout << "Batch test failed: dimension mismatch." << std::endl, exit(-1);
	int correct_answers = 0;
	for (size_t i = 0; i < imgs.size(); ++i)
	{
		if (identify(imgs[i]) == labels[i]) correct_answers++;
	}
	return static_cast<double>(correct_answers) / imgs.size();
}
