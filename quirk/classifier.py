from collections import OrderedDict
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

try:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.utils import np_utils
except ImportError:
    pass

try:
    import xgboost as xgb
except ImportError:
    pass

from .base import Base


class Classifier(Base):
    def _plot_target(self):
        return sns.countplot(self._y())

    def _plot_category(self, name):
        data = self._train_features_df.dropna(subset=[name])
        self._plot(sns.countplot(x=name, data=data))
        self._plot(sns.barplot(x=name, y=self._target_col,
                               data=data))

    def _plot_number(self, name):
        data = self._train_features_df.dropna(subset=[name])
        data = self._drop_outliers(data, name)

        self._plot(sns.distplot(data[name]))
        self._plot(sns.boxplot(y=name, x=self._target_col,
                               data=data))

    def _build_models(self, train_x, train_y, test_x, test_y):
        predictions = OrderedDict()

        if self._eval_metric == 'mlogloss':
            counts = pd.Series(train_y).value_counts()
            prob = counts.apply(lambda x: x / float(counts.sum()))
            row = [prob[k] for k in range(len(self._classes))]
            predictions['Mean'] = pd.DataFrame([row for x in range(len(test_x))])
        else:
            mode = self._mode(train_y)
            predictions['Mode'] = pd.Series([mode] * len(test_x))

        if self._eval_metric != 'mlogloss' and 'Sequential' in globals():
            predictions['Keras'] = self._keras_predict(train_x, train_y, test_x, test_y)
            print(predictions['Keras'])

        predictions['XGBoost'] = self._xgboost_predict(train_x, train_y, test_x, test_y)

        return predictions

    def _keras_predict(self, train_x, train_y, test_x, test_y):
        # https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py
        model = Sequential()
        model.add(Dense(32, input_dim=train_x.shape[1]))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        y = np_utils.to_categorical(train_y)

        history = model.fit(train_x.values, y,
                            nb_epoch=5, batch_size=50,
                            verbose=1, validation_split=0.1)

        return model.predict_classes(test_x.values)

    def _xgboost_predict(self, train_x, train_y, test_x, test_y):
        if self._eval_metric == 'mlogloss':
            params = {
                'seed': self._seed,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': len(self._classes)
            }

            dtrain = xgb.DMatrix(data=train_x, label=train_y)
            dtest = xgb.DMatrix(data=test_x)

            model = xgb.train(params, dtrain, 100, verbose_eval=25)
            self._xgboost_model = model # hack
            return model.predict(dtest)
        else:
            if test_y is None:
                eval_set = []
            else:
                eval_set = [(train_x, train_y), (test_x, test_y)]
            model = xgb.XGBClassifier(seed=self._seed)
            model.fit(train_x, train_y, eval_set=eval_set, eval_metric=(self._eval_metric or 'error'))
            self._xgboost_model = model # hack
            return model.predict(test_x)

    def _score(self, act, pred):
        if self._eval_metric == 'mlogloss':
            return log_loss(act, pred)
        elif self._eval_metric == 'auc':
            return roc_auc_score(act, pred)
        else:
            return accuracy_score(act, pred)
