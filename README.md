## Training My Neural Network with the MNIST Dataset
#### Aakash Pydi
---

The neural network with forward propagation and back-propagation pass is implemented in neural_network.py. The MyImg2Num neural network is defined in my_img2num.py. It relies on the Neural Network class implemented in neural_network.py for its functionality. The NnImg2Num neural network is defined in nn_img2num.py. This neural network uses the torch.nn neural network package for its functionality. Finally both of these neural networks are trained and tested on the MNIST handwritten digits dataset. In order to run the scripts, simply create an instance of the corresponding class and call the train() instance method as shown below.

nn_mnist = NnImg2Num()

nn_mnist.train()

my_mnist = MyImg2Num()

my_mnist.train()

---

The performance charts associated with each of the neural networks are given below.

## NnImg2Num Neural Network

![nn_train_time](/images/train_time_vs_epoch_nn.png)

![nn_test_train](/images/train_vs_test_error_nn.png)

---

## MyImg2Num Neural Network

![my_train_time](/images/train_time_vs_epoch_my.png)

![my_test_train](/images/train_vs_test_error_my.png)

---
