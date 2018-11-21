from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import BatchNormalization
import sklearn.svm as svm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import GRU
from keras import regularizers
from sklearn.base import BaseEstimator
import time

class nueral_network(BaseEstimator):
    """requires some more work"""
    def __init__(self, x_train, y_train, x_test, y_test, nn):
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._nn = nn
        self._model = None

    def create_keras_model(self):
        model = Sequential()
        for layer in self._nn['layers']:
            if layer == 'BatchNormalization':
                model.add(BatchNormalization())
            else:
                for key, value in layer.items():
                    model.add(Dense(units=int(value['nuerons']), activation=value['activation'],\
                        kernel_regularizer=regularizers.l2(float(value['kernel_regularizer']))))
                    model.add(Dropout(float(value['dropout'])))
                    if value['type'] == 'gru':
                        model.add(GRU(10, return_sequences=True))
        if self._nn['optimizer'] == 'Adam':
            adam_optimizer = Adam(lr=self._nn['learning_rate'])
            model.compile(loss=self._nn['loss'], optimizer=self._nn['optimizer'], metrics=self._nn['metric'])
            return model
        return None


    def plot_accuracy(history, attributes=['acc', 'loss']):
        for attr in attributes:
            plt.xlabel('epochs')
            plt.ylabel(attr)
            plt.plot(history.history[attr], label='train')
            plt.plot(history.history['val_' + attr], label='test')
            plt.legend()
            plt.show()


    def fit(self, x_train, y_train, verbose=1):
        self._model = self.create_keras_model()
        history = self._model.fit(x_train, y_train, validation_split=float(self._nn['validation_split']),\
                epochs=int(self._nn['epochs']))
        return self

    def predict(self):
        return self._model.predict_classes(self._x_test)


    def train_mlp_classifier(self):
        model = self.create_keras_model()
        history = model.fit(self._x_train, self._y_train, validation_split=float(self._nn['validation_split']),\
                epochs=int(self._nn['epochs']))
        self.plot_accuracy(history)
        return model.predict_classes(self._x_test), model.predict_classes(self._x_train)


    def train_mlp_regressor(self):
        model = self.create_keras_model()
        history = model.fit(self._x_train, self._y_train, validation_split=float(self._nn['validation_split']),\
                epochs=int(self._nn['epochs']))
        return model.predict(self._x_test), model.predict(self._x_train)


    def rolling_mlp_regressor(self, ori):
        model = self.create_keras_model()
        predictions = []
        errors = []
        for i in range(len(self._x_test)):
            history = model.fit(self._x_train, self._y_train, validation_split=float(self._nn['validation_split']), epochs=int(self._nn['epochs']))
            prediction = model.predict(self._x_test)[i]
            predictions.append(prediction)
            x_train = list(self._x_train)
            y_train = list(self._y_train)
            x_train.append(self._x_test[i])
            y_train.append(self._y_test[i])
            self._x_train = np.array(x_train)
            self._y_train = np.array(y_train)
            errors.append(np.abs(ori[i] * self._y_test[i] - ori[i] * prediction))
            print("mae {}".format(np.abs(ori[i] * self._y_test[i] -  ori[i] * prediction)))
            print("length of ori {}".format(len(ori)))
            print("length of Y_test {}".format(len(self._y_test)))
        plt.ylabel('error in $')
        plt.xlabel('days')
        plt.plot(errors)
        plt.show()
        return np.array(predictions), model.predict(self._x_train)


def svm_regressor(X_train, Y_train, X_test):
    Y_train = Y_train.reshape(len(Y_train),)
    parameters = {
            'kernel' : ['rbf', 'poly'],
            'C' : [100, 500],
            'gamma' : [1e-4],
            'epsilon' : [100, 150]
    }
    svr = svm.SVR()
    clf = GridSearchCV(svr, parameters, n_jobs=6, verbose=10)
    Y_test_pred = clf.fit(X_train, Y_train).predict(X_test)
    Y_train_pred = clf.fit(X_train, Y_train).predict(X_train)
    return Y_pred, Y_train_pred
