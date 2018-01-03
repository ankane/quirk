from collections import OrderedDict
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

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

        # prevent seaborn error
        if(len(data[name].unique())) > 1:
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

        predictions['XGBoost'] = self._xgboost_predict(train_x, train_y, test_x, test_y)
        return predictions

    def _xgboost_predict(self, train_x, train_y, test_x, test_y):
        model = xgb.XGBClassifier(seed=self._seed, n_estimators=100, max_depth=3, learning_rate=0.1)
        self._xgboost_model = model # hack

        eval_metric = self._eval_metric or 'error'

        if test_y is None:
            model.fit(train_x, train_y, eval_metric=eval_metric, verbose=10)
        else:
            eval_set = [(train_x, train_y), (test_x, test_y)]
            model.fit(train_x, train_y, eval_set=eval_set, eval_metric=eval_metric,
                      verbose=10)

        if self._eval_metric == 'mlogloss':
            return model.predict_proba(test_x)
        else:
            return model.predict(test_x)

    def _score(self, act, pred):
        if self._eval_metric == 'mlogloss':
            return log_loss(act, pred)
        elif self._eval_metric == 'auc':
            return roc_auc_score(act, pred)
        else:
            return accuracy_score(act, pred)
