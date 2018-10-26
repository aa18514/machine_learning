from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import BatchNormalization
import sklearn.svm as svm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import yaml
import numpy as np

def load_model(model_file='model\\regressor.yml'):
    cfg = None
    with open(model_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg


def create_keras_model(nn):
    model = Sequential()
    for layer in nn['layers']:
        if layer == 'BatchNormalization':
            model.add(BatchNormalization())
        else:
            for key, value in layer.items():
                model.add(Dense(units=int(value['nuerons']), activation=value['activation']))
                model.add(Dropout(int(value['dropout'])))
    if nn['optimizer'] == 'Adam':
        adam_optimizer = Adam(lr=nn['learning_rate'])
        model.compile(loss=nn['loss'], optimizer=nn['optimizer'], metrics=nn['metric'])
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


def train_mlp_classifier(X_train, Y_train, X_test):
    cfg = load_model('model\\classifier.yml')['nn']
    model = create_keras_model(cfg)
    history = model.fit(X_train, Y_train, validation_split=float(cfg['validation_split']), epochs=int(cfg['epochs']))
    plot_accuracy(history)
    return model.predict_classes(X_test), model.predict_classes(X_train)


def train_mlp_regressor(X_train, Y_train, X_test):
    cfg = load_model()['nn']
    model = create_keras_model(cfg)
    history = model.fit(X_train, Y_train, validation_split=float(cfg['validation_split']), epochs=int(cfg['epochs']))
    return model.predict(X_test), model.predict(X_train)

def rolling_mlp_regressor(X_train, Y_train, X_test, Y_test):
    cfg = load_model()['nn']
    model = create_keras_model(cfg)
    predictions = []
    for i in range(len(X_test)):
        history = model.fit(X_train, Y_train, validation_split=float(cfg['validation_split']), epochs=int(cfg['epochs']))
        print(X_test[i])
        prediction = model.predict(X_test)[i]
        predictions.append(prediction)
        print(X_train.shape)
        X_train = list(X_train)
        Y_train = list(Y_train)
        X_train.append(X_test[i])
        Y_train.append(Y_test[i])
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
    return np.array(predictions), model.predict(X_train)


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
