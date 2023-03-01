from collections import defaultdict
from pathlib import Path
import os
import pickle
import random
import operator
import math
from turtle import distance

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tempfile import TemporaryFile
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from scipy.spatial.distance import euclidean


def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = euclidean(trainingSet[x][0], instance[0])  # use Euclidean distance
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def load_data(directory):
    dataset = []
    i = 0

    for folder in os.listdir(directory):
        print(folder)
        i += 1
        if i == 11:
            break
        for file in os.listdir(directory / folder):
            #print("Processing file:", file)
            try:
                (rate, sig) = wav.read(directory / folder / file)
                mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i)
                dataset.append(feature)
            except ValueError:
                print("Error processing file:", file)

    return dataset


def train_and_evaluate_model(dataset):
    # split data into training and testing sets
    X = [x[0] for x in dataset]
    y = [x[2] for x in dataset]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=105)

    # train the model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # make predictions on the testing set
    y_pred = knn.predict(X_test)

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


directory = Path("C:/Users/Kimberly/PycharmProjects/MusicGGG/Data/genres_original")

# load data
dataset = load_data(directory)

# train and evaluate the model
accuracy = train_and_evaluate_model(dataset)

print("Accuracy:", accuracy)


###########################################


def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


results = defaultdict(int)

i = 1
for folder in os.listdir("C:/Users/Kimberly/PycharmProjects/MusicGGG/Data/genres_original"):
    results[i] = folder
    i += 1

(rate, sig) = wav.read("C:/Users/Kimberly/PycharmProjects/MusicGGG/Data/genres_original/metal/metal.00072.wav")
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature = (mean_matrix, covariance, 0)

pred = nearestClass(getNeighbors(dataset, feature, 5))

print(results[pred])
print(getNeighbors(dataset, feature, 5))