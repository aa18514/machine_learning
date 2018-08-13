import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100
import datetime

def pre_process(x, y):
    x_mod = list(x)
    y_mod = list(y)
    for i in range(len(y)):
        for j in range(1, 4):
            x_mod.append(np.rot90(x[i], j))
            y_mod.append(y[i])
    return np.array(x_mod),\
            np.array(y_mod)

def classification_report_csv(report: classification_report, path: str)->(classification_report, str):
    """
    inspiration taken from: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-
    report-into-csv-tab-delimited-format
    :param report: sklearn classification_report object to be parsed in a csv file
    :param path: path where the csv file should be written to
    :return: void
    """
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('     ') 
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    row_data = lines[-2].split('    ')
    row = {}
    row['class'] = row_data[0]
    row['precision'] = float(row_data[1])
    row['recall'] = float(row_data[2])
    row['f1_score'] = float(row_data[3])
    row['support'] = float(row_data[4])
    report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(path, index=False)


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    return (X_train, y_train), (X_test, y_test)


def olivetti_faces(test_split: int)->int:
    olivetti_faces = fetch_olivetti_faces()
    x, y = olivetti_faces['data'], olivetti_faces['target']
    x = x.reshape(len(y), 64, 64)
    x, y = pre_process(x, y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    #(X_train, y_train), (X_test, y_test) = olivetti_faces(0.20)
    #(X_train, y_train), (X_test, y_test) = load_mnist()
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    X_train = X_train/255
    X_test = X_test/255
    print(X_train.shape)
    print(y_test.flatten())
    clf = ak.ImageClassifier(verbose=True, augment=True)
    start_time = datetime.datetime.now()
    model = clf.fit(X_train, y_train.flatten())
    end_time = datetime.datetime.now()
    print("fit took %f seconds" % (end_time - start_time).total_seconds())
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)
    test_accuracy = np.sum(y_pred == y_test.flatten())
    print("this is test accuracy")
    print(test_accuracy)
    print(len(y_pred))
    test_accuracy = 100 * test_accuracy/len(y_pred)
    print("test accuracy %f: " % test_accuracy)
    report = classification_report(y_test, y_pred)
    classification_report_csv(report, 'classification_report.csv')
