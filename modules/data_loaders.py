import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import TumobrainorDataset


# loading saved training data
with open("./dataset/training_data.pickle", "rb") as f:
    training_data = pickle.load(f)

# loading saved testing data
with open("./dataset/testing_data.pickle", "rb") as f:
    testing_data = pickle.load(f)


# extracting labels and features from training data
labels_list, features_list = [], []
for X, y in training_data:
    labels_list.append(y)
    features_list.append(X)

# 70 % training, 30% validating
X_train, X_valid, y_train, y_valid = train_test_split(
    features_list, labels_list, test_size=0.3, shuffle=True
)

# extracting labels and features from testing data
labels_list, features_list = [], []
for X, y in testing_data:
    labels_list.append(y)
    features_list.append(X)

X_test, y_test = np.array(features_list), np.array(labels_list)

# testing our dataset class
koten_set = TumobrainorDataset(X_valid, y_valid)
koten_loader = DataLoader(koten_set, batch_size=4, shuffle=True, pin_memory=True)


# creating datasets from numpy arrays
train_set = TumobrainorDataset(X_train, y_train)
valid_set = TumobrainorDataset(X_valid, y_valid)
test_set = TumobrainorDataset(X_test, y_test)


# creating data loaders from datasets
train_loader = DataLoader(
    train_set, batch_size=4, shuffle=True, pin_memory=True, drop_last=True
)
valid_loader = DataLoader(
    valid_set, batch_size=4, shuffle=True, pin_memory=True, drop_last=True
)
test_loader = DataLoader(
    test_set, batch_size=4, shuffle=True, pin_memory=True, drop_last=True
)


# testing data loader
X, y = next(iter(test_loader))
print(X.shape, y.shape, y)


# saving all data loaders
with open("./dataset/train_loader.pickle", "wb") as f:
    pickle.dump(train_loader, f)
with open("./dataset/valid_loader.pickle", "wb") as f:
    pickle.dump(valid_loader, f)
with open("./dataset/test_loader.pickle", "wb") as f:
    pickle.dump(test_loader, f)
