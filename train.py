import json
import random
import math
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNetwork
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':

    with open('intents.json', 'r') as f:
        intents = json.load(f)

    allwords = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            allwords.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '.', '!', ',']
    allwords = [stem(w) for w in allwords if w not in ignore_words]
    # Convert to set to avoid duplicates
    allwords = sorted(set(allwords))
    tags = sorted(set(tags))

    print(tags)
    # In x put bag of words
    x_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, allwords)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label) # Dont need one hot encoding

    x_train = np.array(x_train)
    y_train = np.array(y_train)



    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(x_train)
            self.x_data = x_train
            self.y_data = y_train

        #dataset[index]
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    #Hyper parameters

    batch_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    hidden_size = 8
    learning_rate = 0.001
    num_epocs = 1000

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #Our model
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
    #Our loss model and optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoc in range(num_epocs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device, dtype=torch.int64)

            #forward
            outputs = model(words)
            loss = criterion((outputs), (labels))
            #backward + optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoc + 1) % 100 == 0:
            print(f'epoc {epoc+1}/{num_epocs}, loss={loss.item():.4f}')

    print(f'Final loss, loss={loss.item():.4f}')

    data = {
        "model_state" : model.state_dict(),
        "input_size" : input_size,
        "output_size" :output_size,
        "hidden_size" : hidden_size,
        "allwords" : allwords,
        "tags": tags
    }

    File = "data.pth"
    torch.save(data, File)
    print(f'Training Complete and file saved to {File}')
    


