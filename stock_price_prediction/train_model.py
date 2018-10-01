from keras.models import Sequential
from keras.layers import Dense


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

