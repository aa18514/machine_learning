from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import sklearn.svm as svm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import yaml

def create_keras_model(model_file='model\\regressor.yml'):
    cfg = None
    with open(model_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    nn = cfg['nn']
    model = Sequential()
    for layer in nn['layers']:
        layer = list(layer.values())[0]
        model.add(Dense(units=int(layer['nuerons']), activation=layer['activation']))
        model.add(Dropout(int(layer['dropout'])))
    model.compile(loss=nn['loss'], optimizer=nn['optimizer'], metrics=nn['metric'])
    return model


def plot_accuracy(history, attributes=['acc', 'loss']):
    for attr in attributes:
        plt.xlabel('epochs')
        plt.ylabel(attr)
        plt.plot(history.history[attr], label='train')
        plt.plot(history.history['val_' + attr], label='test')
        plt.legend()
        plt.show()


def train_mlp_classifier(X_train, Y_train, X_test):
    model = create_keras_model('model\\classifier.yml')
    history = model.fit(X_train, Y_train, validation_split=0.35, epochs=350)
    plot_accuracy(history)
    return model.predict_classes(X_test), model.predict_classes(X_train)


def train_mlp_regressor(X_train, Y_train, X_test):
    model = create_keras_model()
    history = model.fit(X_train, Y_train, validation_split=0.35, epochs=350)
    return model.predict(X_test), model.predict(X_train)


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
