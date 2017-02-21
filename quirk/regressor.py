from math import sqrt
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

    @staticmethod
    def _benchmark_results(results, dev_y, rows):
        results['Mean'] = pd.Series([dev_y.mean()] * rows)
        results['Median'] = pd.Series([dev_y.median()] * rows)

    @staticmethod
    def _model():
        return xgb.XGBRegressor(seed=2016)

    @staticmethod
    def _score(act, pred):
        return sqrt(mean_squared_error(act, pred))
