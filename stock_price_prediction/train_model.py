from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import sklearn.svm as svm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def create_keras_regressor(neurons=[20, 20, 20, 20, 1],
                 dropout=[0.01, 0.01, 0.01, 0.01, 0.00],
                 activations='relu', 'relu', 'relu', 'relu', 'linear',
                 loss='mse'):
    model = Sequential()
    for i in range(len(dropout)):
        model.add(Dense(units=neurons[i], activation=activations[i]))
        model.add(Dropout(dropout[i]))
    model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
    return model


def create_keras_classifier(neurons=[20, 20, 20, 20, 1],
                 dropout=[0.01, 0.01, 0.01, 0.01, 0.00],
                 activations=['relu', 'relu', 'relu', 'relu', 'sigmoid'],
                 loss='binary_crossentropy'):
    model = Sequential()
    for i in range(len(dropout)):
        model.add(Dense(units=neurons[i], activation=activations[i]))
        model.add(Dropout(dropout[i]))
    model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
    return model


def plot_accuracy(history):
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show()


def train_mlp_classifier(X_train, Y_train, X_test):
    model = create_keras_classifier()
    history = model.fit(X_train, Y_train, validation_split=0.33, epochs=150)
    plot_accuracy(history)
    #X_te = model.predict_proba(X_test)
    X_te[X_te >= 0.50] = 1
    X_te[X_te < 0.50] = 0
    return X_te, model.predict_classes(X_train)


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
