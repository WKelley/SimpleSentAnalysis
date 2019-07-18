# Simple Recurrent Neural Network for Sentiment Analysis

This is a simple RNN model for sentiment analysis. Built using lower-level Torch, the project goal was to illustrate my knowledge of the theory fundamentals behind text
classification and natural language processing. Towards this end, I refrained from using any higher-level wrappers or dependencies, building the RNN from scratch.
I did, however, use Torch for backpropagation and weight optimization to show that I can maneuver within the dependency.

## Model Description

This model is designed to recognize binary sentiment strength ('good' or 'bad') in text input. It takes in word-vectorized sentences and uses an RNN to convert these two
semantics-preserving "sentence vectors" - that is, one vector containing the semantic information of an entire sentence. These sentence vectors are then fed into a shallow
single-layer perceptron, which classifies them into the 'good' or 'bad' sentiment categories. Both the RNN and the single-layer perceptron are then adjusted based on the
resulting loss.

## Resources Used

Dataset: https://www.kaggle.com/bittlingmayer/amazonreviews

The "Amazon Reviews for Sentiment Analysis" dataset is composed of short Amazon reviews and a corresponding "good" or "bad" sentiment label. Dataset is composed of 3.6 million text reviews and labels for training, along with 400 thousand reviews and ratings for testing. (Note: Not all of this data needs to be used if it exceeds computation capabilities - the amount used can be adjusted within the run.py file.)

Word Vectors: https://nlp.stanford.edu/projects/glove/

I used pre-trained GloVe word vectors to word-vectorize text input. These were loaded via the Gensim API. The specific GloVe word vector set used was trained using Twitter corpus data.

## Results

## Dependency Requirements

NumPy >= 1.16.3

Torch >= 1.0.0

Gensim >= 3.8.0

Matplotlib >= 3.0.3

## Usage
Clone inside the folder containing the data files and run the following command:

```bash
python run.py
```

## Project Status and Future Work

As this model is a simple baseline for sentiment analysis tasks, there are several different paths that could be taken for future work. The single-layer perceptron could be
made into a multi-layer perceptron or some other form of densely-connected neural network. A lot could be done with the RNN - the LSTM "black box" is far more common than the
Elman for just about any task, but NLP in particular. While this project is simple, it contains the building blocks for further expansions and tinkering.
