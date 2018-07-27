#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from pandas import read_csv
from sklearn.model_selection import KFold
from pandas import DataFrame
import sys
from sklearn import preprocessing


class logistic_regression:

    def __init__(self, eta, optimizer):
        self.eta = eta
        self.optimizer = optimizer

    def read_input(self, file_path):
        raw = read_csv(file_path, header=None)
        raw.insert(1, 'bias', 1.0)
        x = raw.iloc[:, 1:58]
        y = raw.iloc[:, -1:]
        return np.array(zip(x.values, y.values))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def adam(self, train, batch_size=32, epochs=200, beta_1=0.9, beta_2=0.99, epsilon=1e-8):
        print("======= running adam with learning rate: %.3f" % self.eta)
        X, Y = zip(*train)
        X = np.array(X)
        Y = np.array([y[0] for y in Y])
        num_of_feature = X.shape[1]
        w = np.zeros(num_of_feature)  # for feature weight and bias
        m_t = 0
        v_t = 0
        t = 0
        for epoch in range(1, epochs + 1):
            np.random.shuffle(train)
            for x, y in get_batches(X, Y, X.shape[0] / batch_size):
                Z = np.dot(w, x.T)  # 1 * (n + 1) dot (n + 1) * batch_size
                A = 1.0 / (1.0 + np.exp(-Z))
                dZ = A - y  # 1 * m
                dw = 1 / batch_size * np.matmul(dZ, x)
                t += 1
                m_t = beta_1 * m_t + (1 - beta_1) * dw
                v_t = beta_2 * v_t + (1 - beta_2) * (dw**2)
                mhat = m_t / (1 - beta_1**t)
                vhat = v_t / (1 - beta_2 **t)
                w -= self.eta * mhat / (np.sqrt(vhat) + epsilon)
            progress = (epoch * 1.0 / epochs) * 100
            sys.stdout.write("\r%d%%" % progress)
            sys.stdout.flush()
        sys.stdout.write("\nEnd of optimization!\n")
        sys.stdout.flush()
        return w

    def adagrad(self, train, epochs=2000, batch_size=1, epsilon=1e-8):
        X, Y = zip(*train)
        X = np.array(X)
        Y = np.array([y[0] for y in Y])
        num_of_feature = X.shape[1]
        w = np.zeros(num_of_feature)  # for feature weight and bias
        gti = np.zeros(num_of_feature)
        dw = np.zeros(num_of_feature)
        gti = np.zeros(num_of_feature)
        for epoch in range(1, epochs + 1):
            np.random.shuffle(train)
            for x, y in get_batches(X, Y, X.shape[0] / batch_size):
                Z = np.dot(w, x.T)  # 1 * (n + 1) dot (n + 1) * batch_size
                A = 1.0 / (1.0 + np.exp(-Z))
                dZ = A - y  # 1 * m
                dw = 1 / batch_size * np.matmul(dZ, x)
                gti += dw ** 2
                w -= self.eta * dw / ((gti + epsilon) ** 0.5)
            progress = (epoch * 1.0 / epochs) * 100
            sys.stdout.write("\r%d%%" % progress)
            sys.stdout.flush()
        sys.stdout.write("\nEnd of optimization!\n")
        sys.stdout.flush()
        return w

    def gd(self, train, epoch=200):
        X, Y = zip(*train)
        # X: m * (n + 1), Y: m * 1
        num_of_feature = X.shape[1]
        w = np.zeros(num_of_feature) # 1 * (n+1)
        dw = np.zeros(num_of_feature)
        num_of_example = X.shape[0]
        for i in range(epoch):
            Z = np.dot(w, X.T) # 1 * (n + 1) dot (n + 1) * m
            A = 1.0 / (1.0 + np.exp(-Z))
            dZ = A - Y # 1 * m
            dw = 1/num_of_example * np.matmul(dZ, x)
            w = w - self.eta * dw
            progress = (epoch * 1.0 / epoch) * 100
            sys.stdout.write("\r%d%%" % progress)
            sys.stdout.flush()
        sys.stdout.write("\nEnd of optimization!\n")
        sys.stdout.flush()
        return w

    def sgd(self, train, epoch=200, batch_size=32):
        X, Y = zip(*train)
        num_of_feature = X.shape[1]
        w = np.zeros(num_of_feature)  # for feature weight and bias
        dw = np.zeros(num_of_feature)
        for epoch in range(1, epoch + 1):
            for x, y in get_batches(X, Y, batch_size):
                Z = np.dot(w, x.T)  # 1 * (n + 1) dot (n + 1) * batch_size
                A = 1.0 / (1.0 + np.exp(-Z))
                dZ = A - y  # 1 * m
                dw = 1 / batch_size * np.matmul(dZ, x)
                w = w - self.eta * dw
            progress = (epoch * 1.0 / epoch) * 100
            sys.stdout.write("\r%d%%" % progress)
            sys.stdout.flush()
        sys.stdout.write("\nEnd of optimization!\n")
        sys.stdout.flush()
        return w

    def accuracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def cost(self, data, w, epsilon=1e-10):
        result = 0
        for x, y in data:
            yhat = self.predict(x, w)
            result += -(y * np.log(yhat + epsilon) + (1 - y) * np.log(1 - yhat + epsilon))
        return result

    def predict(self, feature, w):
        yhat = 0
        feature = np.array(feature)
        yhat += w.T.dot(feature)
        return self.sigmoid(yhat)

    def logistic(self, train, test):
        coefs = self.adam(train) if self.optimizer == "adam" else self.adagrad(train)
        predicts = list()
        for data in test:
            yhat = self.predict(data, coefs)
            yhat = round(yhat)
            predicts.append(yhat)
        return predicts

    def evaluate(self, data):
        kf = KFold(n_splits=10)
        scores = list()
        np.random.shuffle(data)
        for train_index, validation_index in kf.split(data):
            train = data[train_index]

            # normalize training data
            train_x = np.array([x for x, y in train])
            train_y = np.array([y for x, y in train])
            # scalar = preprocessing.StandardScaler().fit(train_x)
            # normalized_train_x = scalar.transform(train_x)

            # normalize testing data
            validation = data[validation_index]
            validation_x = np.array([x for x, y in validation])
            # normalized_validation_x = scalar.transform(validation_x)
            validation_y = [y[0] for x, y in validation]

            predictions = self.logistic(zip(train_x, train_y), validation_x)
            score = self.accuracy(validation_y, predictions)
            print("score: %.3f" % score)
            scores.append(score)
        print('Scores: %s with Mean Accuracy: %.3f%% with %.3f' % (scores, (sum(scores) / float(len(scores))), self.eta))

    def normalized(self, train):
        scalar = preprocessing.StandardScaler().fit(train)
        normalized_train = scalar.transform(train)
        return normalized_train, scalar.mean_, scalar.var_

    def dataset_minmax(self, data):
        min_max = list()
        for i in range(len(data[0][0])):
            col_values = list()
            for x, y in data:
                col_values.append(x[i])
            min_value = min(col_values)
            max_value = max(col_values)
            min_max.append((min_value, max_value))
        return min_max

    def normalize_dataset(self, data, mean, standard_deviation):
        return np.array([(x[i] - mean[i]) / standard_deviation[i] for i in range(len(mean)) for x in data])

    def testing(self, filepath, outputpath, eta=0.01):
        test = read_csv(filepath, header=None)
        test_x = test.iloc[:, 1:]
        predicted = self.logistic(dataset, test_x.values)

        result = DataFrame.from_dict(dict({"id": list(range(1, len(predicted) + 1)), "values": predicted}))
        result.to_csv(outputpath, index=False)
        print("Testing done...")


def get_batches(x, y, n_batches=10):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        # If we're not on the last batch, grab data with size batch_size
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
            # On the last batch, grab the rest of the data
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y

if __name__ == '__main__':
    path = './data/spam_train.csv'
    regressor = logistic_regression(0.04, "adagrad")
    dataset = regressor.read_input(path)
    #evaluate(dataset, 0.0000005, "sgd")
    regressor.evaluate(dataset)
    # test_path = '/home/vincent/machine_learning/LogisticRegression/data/spam_test.csv'
    # output_path = "/home/vincent/machine_learning/LogisticRegression/data/test_result.csv"
    # testing(test_path, output_path)
