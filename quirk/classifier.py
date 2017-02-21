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

    @staticmethod
    def _benchmark_results(results, dev_y, rows):
        mode = dev_y.value_counts().index[0]
        results['Mode'] = pd.Series([mode] * rows)

    @staticmethod
    def _model():
        return xgb.XGBClassifier(seed=2016)

    @staticmethod
    def _score(act, pred):
        return accuracy_score(act, pred)
