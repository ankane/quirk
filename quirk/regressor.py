from math import sqrt
from collections import OrderedDict
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

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

    def _build_models(self, dev_x, dev_y, val_x):
        predictions = OrderedDict()
        predictions['Mean'] = pd.Series([dev_y.mean()] * len(val_x))
        predictions['Median'] = pd.Series([dev_y.median()] * len(val_x))
        predictions['XGBoost'] = self._xgboost_predict(dev_x, dev_y, val_x)
        return predictions

    def _xgboost_predict(self, dev_x, dev_y, val_x):
        model = xgb.XGBRegressor(seed=2016)
        model.fit(dev_x, dev_y)
        self._xgboost_model = model # hack
        return model.predict(val_x)

    @staticmethod
    def _score(act, pred):
        return sqrt(mean_squared_error(act, pred))
