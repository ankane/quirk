from math import sqrt
from collections import OrderedDict
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from numpy import log1p

try:
    import xgboost as xgb
except ImportError:
    pass

from .base import Base


class Regressor(Base):
    def _plot_target(self):
        return sns.distplot(self._y())

    def _plot_category(self, name):
        data = self._train_features_df.dropna(subset=[name])
        self._plot(sns.countplot(x=name, data=data))
        self._plot(sns.boxplot(x=name, y=self._target_col,
                               data=data))

    def _plot_number(self, name):
        data = self._train_features_df.dropna(subset=[name])
        self._plot(sns.distplot(data[name]))
        self._plot(sns.regplot(x=name, y=self._target_col,
                               data=data))

    def _build_models(self, train_x, train_y, test_x, test_y):
        predictions = OrderedDict()
        predictions['Mean'] = pd.Series([train_y.mean()] * len(test_x))
        predictions['Median'] = pd.Series([train_y.median()] * len(test_x))
        predictions['XGBoost'] = self._xgboost_predict(train_x, train_y, test_x, test_y)
        return predictions

    def _xgboost_predict(self, train_x, train_y, test_x, test_y):
        model = xgb.XGBRegressor(seed=self._seed, n_estimators=300, max_depth=3, learning_rate=0.1)
        self._xgboost_model = model # hack

        if test_y is None:
            model.fit(train_x, train_y, verbose=10)
        else:
            eval_set = [(train_x, train_y), (test_x, test_y)]
            model.fit(train_x, train_y, eval_set=eval_set,
                      early_stopping_rounds=25, verbose=10)

        return model.predict(test_x)

    def _score(self, act, pred):
        if self._eval_metric == 'rmsle':
            return sqrt(mean_squared_error(log1p(act), log1p(pred)))
        else:
            return sqrt(mean_squared_error(act, pred))
