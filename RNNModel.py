import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#Weight and Layer Initialization Methods

def initialize_parameters(wordvec_dim, output_dim, hidden_dim):

    #Initialize Parameters to Random Values
    input_weights = torch.rand([hidden_dim, wordvec_dim], dtype = torch.float32, requires_grad = True)
    hidden_weights = torch.rand([hidden_dim, hidden_dim], dtype = torch.float32, requires_grad = True)
    output_weights = torch.rand([output_dim, hidden_dim], dtype = torch.float32, requires_grad = True)
    bias = torch.rand(hidden_dim, dtype = torch.float32, requires_grad = True)

    #Build a Parameter Dictionary
    parameters = {}
    parameters['input_weights'] = input_weights
    parameters['hidden_weights'] = hidden_weights
    parameters['output_weights'] = output_weights
    parameters['bias'] = bias

    return parameters

def initialize_hidden(shape):

    #Initialize Hidden Layer Vector
    return torch.zeros(shape, dtype = torch.float32, requires_grad = False)



#RNN Structure Methods

def blackbox_elman(word_vector, hidden_prev, input_weights, hidden_weights, bias):

    #Elman Computation
    inside = torch.sigmoid(torch.matmul(input_weights, word_vector) + torch.matmul(hidden_weights, hidden_prev) + bias)
    return torch.sigmoid(inside)

def classification_perceptron(hidden, output_weights):

    #Perceptron Computation
    softmax = nn.Softmax(dim = 0)
    layer = torch.matmul(output_weights, hidden)
    return softmax(layer)

def network_forward(sentence_vectors, parameters):

    #Read Weights from Parameter Dictionary
    input_weights = parameters['input_weights']
    hidden_weights = parameters['hidden_weights']
    output_weights = parameters['output_weights']
    bias = parameters['bias']

    #From Word Vectors, Use RNN to Compute Sentence Vector
    hidden_prev = initialize_hidden(len(hidden_weights[1]))
    for i in range(len(sentence_vectors)):
        word_vector = sentence_vectors[i]
        hidden_current = blackbox_elman(word_vector, hidden_prev, input_weights, hidden_weights, bias)
        hidden_prev = hidden_current

    #Run Classification Based on Sentence Vector
    sentiment_prediction = classification_perceptron(hidden_current, output_weights)
    return sentiment_prediction



#Loss Function

def loss_function(predicted, actual):

    #Use Actual and Predicted Values to Compute Loss (Mean Squared Error)

    #'actual' is the true value of a binary classification, either 1 or 0.
    #'predicted' is a two-dimensional list of probabilities corresponding to the
    #likelihood that either class is correct.

    #A probability array of [1 0] is the ideal prediction for an actual value of 1,
    #as the first number is the likelihood that the actual value will be 1,
    #while the second number is the likelihood that the actual value will be 0.
    #Conversely, [0 1] is the ideal prediction for an actual value of 0.

    loss = 0
    if actual == 1:
        loss += (1 - predicted[0]) ** 2
        loss += (0 - predicted[1]) ** 2
    elif actual == 0:
        loss += (0 - predicted[0]) ** 2
        loss += (1 - predicted[1]) ** 2
    return loss
