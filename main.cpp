#include "mnist_reader.h"
#include "network.h"

int main()
{
	create_instructor ci;
	ci.activation = sigmoid;
	ci.d_activation = d_sigmoid;
	ci.perception_neurons = 28 * 28;
	ci.hidden_layers = { 150 };
	ci.output_neurons = 10;

	mnist_reader mr_train("C:\\Users\\mihai\\Desktop\\progy\\C & C++\\Digits_Identifier\\Data", "train_labels.txt", "train-images.idx3-ubyte");
	mnist_reader mr_test("C:\\Users\\mihai\\Desktop\\progy\\C & C++\\Digits_Identifier\\Data", "test_labels.txt", "t10k-images.idx3-ubyte");

	std::vector<digit_container> train_imgs, test_imgs;
	std::vector<int> train_labels, test_labels;

	train_imgs = mr_train.read_images(60'000);
	train_labels = mr_train.read_labels(60'000);
	test_imgs = mr_test.read_images(1'000);
	test_labels = mr_test.read_labels(1'000);
	
	network N(ci);

	learn_instructor li;
	/* ======================================================================================= */
	li.batch_size = 10;        li.L2_lambda = 0.0;               li.train_answers = train_labels;
	li.dynamic_LR = true;      li.mult_factor = 0.85;            li.test_images = test_imgs;
	li.epoch_count = 20;       li.update_frequency = 5;          li.test_answers = test_labels;
	li.epoch_logs = true;	   li.train_images = train_imgs;     li.learning_rate = 3.0;
	li.thread_number = 4;
	/* ======================================================================================= */

	N.sgd_parallel_learn(li);
	std::cout << "Learning done." << std::endl;
	N.store("C:\\Users\\mihai\\Desktop\\progy\\Additional\\Neural_Networks_Dump\\Digits_Identifier\\Network_1");

	//N.load("C:\\Users\\mihai\\Desktop\\progy\\Additional\\Neural_Networks_Dump\\Digits_Identifier\\Network_1", 2);

	std::cout << "Accuracy: " << N.batch_test(test_imgs, test_labels) << std::endl;

	return 0;
}
