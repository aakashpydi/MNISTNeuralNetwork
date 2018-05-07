import numpy as np
import torch
import math

class NeuralNetwork:

    # Create the dictionary of matrixes Theta (layer). provides
    # theta weight value from layer to layer
    def __init__(self, network_layers):
        #network layers gives no. of nodes in layers,
        #input layer --> hidden layer_1 --> hidden layer_2 --> ... --> output layer
        self.total_error_neural_network = 100.0
        #need to initialize theta dictionary
        self.network_layers = network_layers
        self.theta = dict()

        self.actual_sigmoid_output = dict()
        self.input_mult_theta_output = dict()
        self.output_mult_error = dict()

        self.layer_count = len(network_layers)

        self.edges_keys = ["" for x in range(len(self.network_layers) - 1)]

        for i in range(len(self.network_layers) - 1):
            self.edges_keys[i] = "from:" +str(i)+"--"+"to:"+str(i+1)


        for i in range(len(self.network_layers) - 1):
            #need to initalize theta dictionary values. with mean 0 and std dev <- 1/sqrt(layer_size)
            nodes_in_current_layer = network_layers[i] + 1 #add bias node to each layer
                                                                    #Bias nodes are added to feedforward neural networks to help them learn patterns.
                                                                    #Bias nodes function like an input node that always produces constant. THe constant is called bias activation
            nodes_in_next_layer = network_layers[i + 1]             #note bias nodes have no input nodes
            size = (nodes_in_current_layer, nodes_in_next_layer)
            random = np.random.normal(0, 1/math.sqrt(self.network_layers[i]),size)

            self.theta.update({self.edges_keys[i]:torch.from_numpy(random).float()}) #update dictionary
        #print self.theta

    def getLayer(self, layer_requested):
        return self.theta[self.edges_keys[layer_requested]]


    def forward(self, input_tensor):
        #print "entering forward."
        #print input_tensor
        tensor_passed_through_network = input_tensor.t()
        #print tensor_passed_through_network
        #print input_tensor.size()
        (x, y) = tensor_passed_through_network.size()
        bias_node = torch.ones((1, y)) #returns tensor with dimension 1 * y
        bias_node = bias_node.type(torch.FloatTensor) #in case input is a 1d FloatTensor

        for i in range(len(self.network_layers) - 1):
            tensor_passed_through_network = torch.cat((bias_node, tensor_passed_through_network), dim=0) #concatenated along dimension 0
            self.actual_sigmoid_output.update({i:tensor_passed_through_network})

            layer_weights_transposed = torch.t(self.theta[self.edges_keys[i]])
            #print layer_weights_transposed
            #print tensor_passed_through_network
            results = torch.mm(layer_weights_transposed, tensor_passed_through_network)
            self.input_mult_theta_output.update({i+1:results})
            # need to use sigmoid on results to see what values propagated forward
            # results_np = results.numpy()
            # results_np = 1/(1 + np.exp(-results_np))
            # results = torch.from_numpy(results_np)
            tensor_passed_through_network = torch.sigmoid(results)
            self.actual_sigmoid_output.update({i+1:tensor_passed_through_network})

        #tensor_passed_through_network = torch.transpose(input_tensor, 0, 1)
        #print "returning: "+str(tensor_passed_through_network.t())
        return tensor_passed_through_network.t()

    def backward(self, target_output):
        #print 'entering backward:'
        target_output = target_output.t()
        #print "target output:" + str(target_output)
        #self.error_a =  ((self.a[self.L] - self.target).pow(2).sum())/2
        (x, y) = target_output.size()
        #print target_output.size()
        #print self.actual_sigmoid_output[self.layer_count - 1].size()
        self.total_error_neural_network = (target_output - self.actual_sigmoid_output[self.layer_count - 1]).pow(2).sum()/(2.0 * y)  #(1/2)*(target_output - actual_output) ^ 2
        #print "TOTAL ERROR NEURAL NETWORK\t" + str(self.total_error_neural_network)

        #sigmoid function derivative given by --> derivative = output(1.0 - output)
        def find_sig_derivative(sigmoid_output):
            return sigmoid_output*(1.0 - sigmoid_output)

        # error for neuron = (expected - output) * find_sig_derivative(output)
        derivative_output_layer = find_sig_derivative(self.actual_sigmoid_output[self.layer_count - 1])
        #print "derivative_output_layer: \t"+str(derivative_output_layer)
        error = torch.mul(self.actual_sigmoid_output[self.layer_count - 1] - target_output, derivative_output_layer)
        #print "error: " + str(error)

        for i in range(self.layer_count-2, -1, -1):
            #print i
            #print self.layer_count
            # error = (weight_k * error_j) * derivative(output) where error_j --> error signal from jth neuron, weight_k weight that connects
            #kth neuron to the current neuron
            if i == self.layer_count-2:
                self.output_mult_error.update({i: torch.mm(error, self.actual_sigmoid_output[i].t())})
                # print "multiplying"
                # print i
                # print "actual:\t"+str(self.actual_sigmoid_output[i])
                # print "error:\t"+str(error.t())
                # print "result: \t" + str(self.output_mult_error)
            else:
                #print self.actual_sigmoid_output[i]
                #print "ERROR CHECK 1: \t" + str(error)
                error = error.narrow(0, 1, error.size(0) - 1)                  #torch.index_select(error, 0, torch.LongTensor([1, 2]))

                #print "ERROR CHECK 2: \t" + str(error)
                self.output_mult_error.update({i:torch.mm(error, self.actual_sigmoid_output[i].t())})
                #print "ERROR CHECK 3: \t" + str(error)

            derivative_current_layer = find_sig_derivative(self.actual_sigmoid_output[i])
            #print "DEBUG:\t" + str(self.edges_keys)
            #print str(i)+"\tTHETA:\t" +str(self.theta[self.edges_keys[i]])
            #print str(i)+"\tPRE-Error:\t" +str(error)
            temp = self.theta[self.edges_keys[i]].mm(error)
            error = torch.mul(temp, derivative_current_layer)
            #print str(i)+"\tPOST-Error:\t" +str(error)

    def updateParams(self, learning_rate):
        #weight = weight - learning_rate*error*input
        #print "in UPDATE PARAMS"
        for i in range(self.layer_count - 1):
            temp = torch.mul(self.output_mult_error[i], learning_rate)
            #print "weight update:\t"+str(temp)
            #print "old weight: \t" +str(self.theta[self.edges_keys[i]])
            #print self.theta[self.edges_keys[i]].size()
            #print temp.size()
            self.theta[self.edges_keys[i]] = self.theta[self.edges_keys[i]] - temp.t()
            #print "new weight: \t" +str(self.theta[self.edges_keys[i]])
