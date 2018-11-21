from sklearn import preprocessing


class pre_processor:
    def __init__(self, string_pre_processor):
        self._string_pre_processor = string_pre_processor


    def fit(self, x_train, y_train=None):
        if self._string_pre_processor == 'z_score':
            self._transformer = preprocessing.StandardScaler()
            self._train_data = self._transformer.fit(x_train).\
                    transform(self._train_data)
        else:
            self._transformer = preprocessing.MinMaxScaler()
            self._train_data = self._transformer.fit(x_train).\
                    transform(self._train_data)
        return self


    def transform(self, x_train, y_train=None):
      return self._transformer.transform(x_train)
