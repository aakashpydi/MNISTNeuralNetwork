from NeuralNetwork import NeuralNetwork
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt

class MyImg2Num:

    def __init__(self):

        self.learning_rate = 0.08
        #Images are 28 x 28
        self.image_x = 28
        self.image_y = 28

        self.label_count = 10 #the ten digits {0,1,..,9}

        input_layer_node_count = self.image_x * self.image_y #784
        hidden_layer_1_node_count = 256
        hidden_layer_2_node_count = 128
        output_layer_node_count = self.label_count

        self.my_img_2_num_model = NeuralNetwork([input_layer_node_count, hidden_layer_1_node_count, hidden_layer_2_node_count, output_layer_node_count])


    def train(self):

        #print "In MyImg2Num Train"
        train_batch_size = 50
        mnist_train_data = datasets.MNIST(root='./train_data_set', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        training_data_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=train_batch_size, shuffle=True)

        test_batch_size = 500
        mnist_test_data = datasets.MNIST(root='./test_data_set', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_data_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=test_batch_size, shuffle=True)

        training_time_list = list()
        training_error_list = list()
        test_error_list = list()
        epoch_list = list()

        max_epochs = 45
        epochs = 0
        print "starting epochs"
        while epochs < max_epochs:
            ################################## Training ####################################
            #iterate over whole dataset
            train_error = 0.0
            training_start_time = time.time()
            for (input_data, target_tuple) in training_data_loader:
                #print input_data
                #print target_tuple
                #Need to do one hot encoding of target_tuple
                oneHot_target = torch.zeros(train_batch_size, self.label_count) #10 different columns for each label. set to 1 or 0 to indicate presence
                for i in range(train_batch_size):
                    oneHot_target[i][target_tuple[i]] = 1 #remember target_tuple[i] gives us class label {0,..,9}. So we set corresponding class label to 1

                #neural net foward pass
                feed_forward_results = self.my_img_2_num_model.forward(input_data.view(train_batch_size, self.image_x * self.image_y))

                #neural backward pass
                #print oneHot_target
                self.my_img_2_num_model.backward(torch.FloatTensor(oneHot_target))

                train_error += self.my_img_2_num_model.total_error_neural_network

                #neural net update params
                self.my_img_2_num_model.updateParams(self.learning_rate)

            training_time = time.time() - training_start_time
            training_time_list.append(training_time)

            #print "Train_error: \t" + str(train_error)
            train_error = (train_error)/(len(training_data_loader.dataset)/train_batch_size)
            training_error_list.append(train_error)

            epoch_list.append(epochs+1)
            print "EPOCH: "+str(epochs+1)+"\n\tTraining Error: "+str(train_error)
            ################################## Testing ####################################
            correct_count = 0
            test_error = 0.0

            for (input_data, target_tuple) in test_data_loader:
                #Need to do one hot encoding of target_tuple
                oneHot_target = torch.zeros(test_batch_size, self.label_count) #10 different columns for each label. set to 1 or 0 to indicate presence
                for i in range(test_batch_size):
                    oneHot_target[i][target_tuple[i]] = 1 #remember target_tuple[i] gives us class label {0,..,9}. So we set corresponding class label to 1


                feed_forward_results = self.my_img_2_num_model.forward(input_data.view(test_batch_size, self.image_x * self.image_y))

                #need to calculate as error only updated when we use backward propagation
                loss = (0.5)*((oneHot_target - feed_forward_results).pow(2).sum()) #MSE formula

                test_error += loss

                label, label_index = torch.max(feed_forward_results, 1)
                for i in range(test_batch_size):
                    #print "Label:\t"+str(label_index[i])
                    #print  "Target:\t"+str(target_tuple[i])
                    if label_index[i] == target_tuple[i]:
                        #print "correct count incremented"
                        correct_count += 1
            test_error = (test_error)/(len(test_data_loader.dataset))
            print "\tTest Error: \t" + str(test_error)
            test_error_list.append(test_error)
            #print "correct_count: \t" + str(correct_count)
            #print len(test_data_loader.dataset)
            print "\tAccuracy: \t" + str(correct_count/float(len(test_data_loader.dataset)))
            print "------\n\n"
            epochs += 1

        # GRAPH 1:
        plt.plot(epoch_list, training_time_list)
        plt.xlabel('Epochs')
        plt.ylabel('Training Time')
        plt.title('My Image to Num: Training Time vs Epochs')
        plt.grid(True)
        plt.show()

        # GRAPH 2
        plt.plot(epoch_list, training_error_list, label='Training Error')
        plt.plot(epoch_list, test_error_list, label='Test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('My Image to Num: Error vs Epochs')
        plt.grid(True)
        plt.legend()
        plt.show()



    def forward(self, input_image): #will be 28 x 28 ByteTensor
        #print "In NnImg2Num Forward"
        feed_forward_results = self.my_img_2_num_model.forward(input_image.view(1, self.image_x * self.image_y))
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return class_label
