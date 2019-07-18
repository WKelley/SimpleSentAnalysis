import numpy as np
import bz2
import os
import re
from gensim.models import Word2Vec
import gensim.downloader as api
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from RNNModel import *

#Hyperparameters
NUM_EPOCHS = 2500
LEARNING_RATE = 0.005
HIDDEN_LAYER_DIMENSION = 10

#Immutable Global Parameters
WORD_VECTOR_LENGTH = 25
OUTPUT_LAYER_DIMENSION = 2

#Data Usage
#Can be increased up to the entirety of the dataset. Choose values that fit the
#computational power of your machine.
TRAIN_DATA_POINTS = 4500
TEST_DATA_POINTS = 500



def load_data():

    #Read in Test and Train Files
    train_file = bz2.BZ2File('train.ft.txt.bz2')
    test_file = bz2.BZ2File('test.ft.txt.bz2')

    #Read Data Off of Files
    train_file_lines = train_file.readlines()
    train_file_lines = train_file_lines[0:TRAIN_DATA_POINTS]
    test_file_lines = test_file.readlines()
    test_file_lines = test_file_lines[0:TEST_DATA_POINTS]

    #Convert from Binary Strings to Parsable Strings
    train_file_lines = [x.decode('utf-8') for x in train_file_lines]
    test_file_lines = [x.decode('utf-8') for x in test_file_lines]

    #Extract Training Text and Sentiment Labels
    train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
    train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

    #Extract Test Text and Sentiment Labels
    test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
    test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]

    return train_labels, train_sentences, test_labels, test_sentences



def preprocess_data(train_sentences, test_sentences):

    #Remove Numbers from Sentences
    for i in range(len(train_sentences)):
        train_sentences[i] = re.sub('\d', '', train_sentences[i])
    for i in range(len(test_sentences)):
        test_sentences[i] = re.sub('\d', '', test_sentences[i])

    #Replace URLs in Sentences with Generic <URL> Tag
    for i in range(len(train_sentences)):
        if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
            train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
    for i in range(len(test_sentences)):
        if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
            test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

    return train_sentences, test_sentences



def sentence_word_vectorization(train_sentences, test_sentences):

    #Initialize arrays that will contain word-vectorized versions of sentences.
    word_vectorized_train_sentences = []
    word_vectorized_test_sentences = []

    #Load Pre-Trained Word Vector Model from GloVe
    model = api.load("glove-twitter-25")

    #Word-Vectorize Sentences, Add to Vectorized Sentence Arrays
    for i in range(len(train_sentences)):
        word_vectorized_train_sentence = [torch.FloatTensor(model[word]) for word in (train_sentences[i].split()) if word in model]
        word_vectorized_train_sentences.append(word_vectorized_train_sentence)
    for i in range(len(test_sentences)):
        word_vectorized_test_sentence = [torch.FloatTensor(model[word]) for word in (test_sentences[i].split()) if word in model]
        word_vectorized_test_sentences.append(word_vectorized_test_sentence)

    return word_vectorized_train_sentences, word_vectorized_test_sentences



def main():

    #Load in Data
    train_labels, train_sentences, test_labels, test_sentences = load_data()

    #Pre-Process Data
    train_sentences, test_sentences = preprocess_data(train_sentences, test_sentences)

    #Word-Vectorize Sentence Text
    word_vectorized_train_sentences, word_vectorized_test_sentences = sentence_word_vectorization(train_sentences, test_sentences)

    #Parameter and Optimizer Initialization
    parameters = initialize_parameters(WORD_VECTOR_LENGTH, OUTPUT_LAYER_DIMENSION, HIDDEN_LAYER_DIMENSION)
    optimizer = optim.SGD(list(parameters.values()), lr = LEARNING_RATE)

    #Initialize Lists to Record Loss and Test Accuracy
    accuracy_record = []
    loss_record = []

    for epoch in range(NUM_EPOCHS):

        #Read Out Current Epoch
        print('Epoch:', epoch + 1)

        #Training Loop
        total_loss = 0
        for sentence in range(TRAIN_DATA_POINTS):

            #Clean Optimizer, Calculate Predictions
            optimizer.zero_grad()
            predictions = network_forward(word_vectorized_train_sentences[sentence], parameters)

            #Calculate Loss, Add to Total
            loss = loss_function(predictions, train_labels[sentence])
            total_loss += loss

            #Backpropagation and Weight Adjustment
            loss.backward()
            optimizer.step()

        #Read Out Loss for Every Epoch, Record on List
        loss_record.append(total_loss.item())
        print('Training Loss:', total_loss.item())

        #Testing Loop
        total_correct_predictions = 0
        for sentence in range(TEST_DATA_POINTS):

            #Calculate Predicted Class of Testing Data
            predictions = network_forward(word_vectorized_test_sentences[sentence], parameters)

            #Determine Accuracy of Predictions
            if predictions[0] > predictions[1] and test_labels[sentence] == 1:
                total_correct_predictions += 1
            elif predictions[0] < predictions[1] and test_labels[sentence] == 0:
                total_correct_predictions += 1

        #Read Out Testing Accuracy for Every Epoch, Record on List
        test_accuracy = total_correct_predictions / TEST_DATA_POINTS
        accuracy_record.append(test_accuracy)
        print('Test Accuracy:', test_accuracy)

        #Space Read-Outs Between Epochs
        print('')

    #Read Out Graph of Test Accuracy Over Time
    plt.plot(accuracy_record)
    plt.ylabel('Epoch')
    plt.xlabel('Testing Accuracy')
    plt.show()

    #Read Out Graph of Loss Over Time
    plt.plot(loss_record)
    plt.ylabel('Epoch')
    plt.xlabel('Training Loss')
    plt.show()


if __name__ == '__main__':
    main()
