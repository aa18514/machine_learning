from keras.models import Sequential
from keras.layers import Dense
import sklearn.svm as svm


def create_model(neurons=[30, 30, 30, 30, 1],
                 dropout=[0.10, 0.10, 0.10, 0.10, 0.00],
                 activations=['tanh', 'tanh', 'tanh', 'tanh', 'sigmoid'],
                 loss='binary_crossentropy'):
    model = Sequential()
    for i in range(len(dropout)):
        model.add(Dense(units=neurons, activation=activations[i]))
        model.add(Dropout(dropout[i]))
    model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
    return model


def train_mlp_classifier(X_train, Y_train, X_test):
    model = KerasClassifier(build_fn=create_model, epochs=150, verbose=70)
    model = create_model()
    model.fit(X_train, Y_train, epochs=250)
    return model.predict_classes(X_test), model.predict_classes(X_train)


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
