from collections import OrderedDict
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

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
        mode = train_y.value_counts().index[0]
        predictions['Mode'] = pd.Series([mode] * len(test_x))
        predictions['XGBoost'] = self._xgboost_predict(train_x, train_y, test_x, test_y)
        return predictions

    def _xgboost_predict(self, train_x, train_y, test_x, test_y):
        model = xgb.XGBClassifier(seed=2016)
        if test_y is None:
            eval_set = []
        else:
            eval_set = [(train_x, train_y), (test_x, test_y)]
        model.fit(train_x, train_y, eval_set=eval_set)
        self._xgboost_model = model # hack
        return model.predict(test_x)

    @staticmethod
    def _score(act, pred):
        return accuracy_score(act, pred)
