import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence ,allwords):
    sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(allwords), dtype= np.float32)
    for i, w in enumerate(allwords):
        if w in sentence:
            bag[i] = 1
    return bag

