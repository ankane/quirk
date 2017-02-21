from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import __main__

# analyze
import numpy as np
import pandas as pd
import seaborn as sns

# model
from sklearn import model_selection, preprocessing

try:
    import xgboost as xgb
except ImportError:
    pass

# screen
from .notebook import Notebook
from .terminal import Terminal


class Base(object):
    __metaclass__ = ABCMeta

    def __init__(self, train_data=None, test_data=None, target_col=None,
                 id_col=None, datetime_cols=None):

        train_df = self._fetch_data(train_data)
        test_df = self._fetch_data(test_data)

        # default is None rather than []
        # due to potentially dangerous Python behavior
        if datetime_cols is not None:
            for col in datetime_cols:
                train_df[col] = pd.to_datetime(train_df[col])
                if test_df is not None:
                    test_df[col] = pd.to_datetime(test_df[col])

        # store preview of correctly ordered columns for later
        self._train_df_head = train_df.head()

        # reorder column
        y = train_df[target_col]
        train_df.drop(target_col, axis=1, inplace=True)
        train_df = pd.concat([train_df, y], axis=1)

        # store
        self._train_df = train_df
        self._test_df = test_df
        self._target_col = target_col
        self._id_col = id_col

        self._train_features_df = None
        self._test_features_df = None

        if self._is_interactive():
            self._screen = Notebook()
        else:
            self._screen = Terminal()

    def analyze(self):
        self._summarize_data()
        self._show_target_variable()
        self._show_pairwise_correlation()
        self._show_features()

    def model(self):
        if 'xgb' not in globals():
            raise ValueError('XGBoost not installed')

        self._header('Model')

        target_col = self._target_col
        id_col = self._id_col

        if self._train_features_df is None:
            self._generate_features(viz=False)

        train_model_df = self._train_features_df.drop(target_col, axis=1)
        test_model_df = self._test_features_df.copy()

        # make numeric
        # TODO one hot encoding for linear regression
        for f in train_model_df.columns:
            if train_model_df[f].dtype == 'object':
                lbl = preprocessing.LabelEncoder()
                train_model_df[f] = lbl.fit_transform(train_model_df[f].fillna('?'))
                if self._test_df is not None:
                    test_model_df[f] = lbl.fit_transform(test_model_df[f].fillna('?'))

        dev_x, val_x, dev_y, val_y = model_selection.train_test_split(
            train_model_df, self._y(), test_size=0.33, random_state=2016)

        predictions = OrderedDict()

        # benchmark models
        self._benchmark_results(predictions, dev_y, len(val_x))

        # xgboost
        model = self._model()
        model.fit(dev_x, dev_y)
        predictions['XGBoost'] = model.predict(val_x)

        # score models
        scores = OrderedDict()
        for k, v in predictions.items():
            scores[k] = self._score(val_y, v)

        # comparison
        self._subheader('Comparison')
        # odict hack for Python 3
        self._plot(sns.barplot(x=list(scores.keys()), y=list(scores.values())))

        # show xgboost info
        self._subheader('XGBoost')
        self._paragraph('RMSE: %f' % scores['XGBoost'])
        self._plot(xgb.plot_importance(model))

        # save data
        if self._test_df is not None:
            self._header('Test Predictions')

            # retrain model on all data
            final_model = self._model()
            final_model.fit(train_model_df, self._y())

            preds = final_model.predict(test_model_df)
            out_df = pd.DataFrame()
            out_df[id_col] = self._test_df[id_col].values
            out_df[target_col] = preds

            filename = 'results.csv'
            out_df.to_csv(filename, index=False)
            self._table(out_df.head())
            self._paragraph('Saved to %s' % filename)

    @staticmethod
    def _fetch_data(data):
        if data is None:
            return None
        elif isinstance(data, str):
            if data.endswith('.json'):
                return pd.read_json(data)
            else:
                return pd.read_csv(data)
        else:
            return data.copy()

    def _summarize_data(self):
        self._header('Data')

        lines = ['Training set: %d rows' % len(self._train_df)]
        if self._test_df is not None:
            lines.append('Test set: %d rows' % len(self._test_df))

        self._paragraph(*lines)
        self._table(self._train_df_head)

    def _show_target_variable(self):
        self._header('Target Variable')
        self._plot(self._plot_target())

    def _show_pairwise_correlation(self):
        self._header('Pairwise Correlation')
        corr = self._train_df.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        self._plot(sns.heatmap(corr.abs(), mask=mask, cmap='OrRd'))

    def _show_features(self):
        self._header('Columns')
        self._generate_features()

    # collect info
    # generate features
    # display features
    def _generate_features(self, viz=True):
        target_col = self._target_col

        self._train_features_df = pd.DataFrame()
        self._test_features_df = pd.DataFrame()
        self._train_features_df[target_col] = self._train_df[target_col]

        for name, col in self._train_df.iteritems():
            if name == target_col:
                continue

            info = self._col_info(col)
            unique_count = info['unique_count']
            null_count = info['null_count']
            column_type = info['column_type']

            if viz:
                self._subheader(col.name)

                lines = ['Type: %s' % column_type, 'Unique values: %s' %
                                unique_count,
                                self._show_pct('Missing values: %s', null_count, info['total_count'])]

                if column_type == 'number':
                    outliers = info['total_count'] - null_count - len(self._drop_outliers(self._train_df.dropna(subset=[col.name]), col.name))
                    lines.append(self._show_pct('Outliers: %d', outliers, info['total_count']))

                self._paragraph(*lines)

            # add features
            if column_type == 'number' or column_type == 'category':
                self._train_features_df[col.name] = self._train_df[col.name]
                if self._test_df is not None:
                    self._test_features_df[col.name] = self._test_df[col.name]
            elif column_type == 'time':
                self._train_features_df[col.name + '_hour'] = col.dt.hour
                self._train_features_df[col.name + '_weekday'] = col.dt.weekday
                if self._test_df is not None:
                    self._test_features_df[
                        col.name + '_hour'] = self._test_df[col.name].dt.hour
                    self._test_features_df[
                        col.name + '_weekday'] = \
                        self._test_df[col.name].dt.weekday

            # visualize
            if viz:
                if column_type == 'category' and unique_count <= 20:
                    self._plot_category(col.name)
                elif column_type == 'number':
                    self._plot_number(col.name)
                elif column_type == 'time':
                    self._plot_category(col.name + '_hour')
                    self._plot_category(col.name + '_weekday')

        self._show_geo()

    @staticmethod
    def _show_pct(message, num, denom):
        message = message % num
        if num > 0:
            pct = 100.0 * num / denom
            message += ' (%.1f%%)' % round(pct, 1)
        return message

    def _show_geo(self):
        train_df = self._train_df

        geo_cols = ['latitude', 'longitude']

        # latitude and longtitude
        if all(x in train_df.keys() for x in geo_cols):
            self._subheader("latitude & longitude")

            # drop outliers
            data = train_df.dropna(subset=geo_cols)
            for col in geo_cols:
                data = self._drop_outliers(data, col)

            # TODO use map
            self._plot(sns.lmplot(y='latitude', x='longitude', hue='interest_level', fit_reg=False, data=data))

            # TODO geohashes

    @staticmethod
    def _drop_outliers(data, name):
        # TODO check dataset skew
        quartile_1, quartile_3 = np.percentile(data[name], [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return data[(data[name] >= lower_bound) & (data[name] <= upper_bound)]

    def _y(self):
        return self._train_df[self._target_col]

    @staticmethod
    def _col_info(col):
        total_count = len(col)

        try:
            unique_count = len(col.unique())
        except TypeError:
            unique_count = len(col.apply(tuple).unique())

        null_count = col.isnull().sum()

        numeric = col.dtype == 'int64' or col.dtype == 'float64'
        if col.dtype == 'datetime64[ns]':
            column_type = 'time'
        elif unique_count / float(total_count) > 0.95:
            column_type = 'unique'
        elif col.name.endswith('id') or unique_count <= 20:
            column_type = 'category'
        elif numeric:
            column_type = 'number'
        else:
            column_type = 'unknown'

        return {'total_count': total_count, 'unique_count': unique_count,
                'null_count': null_count, 'column_type': column_type}

    # test interactive

    @staticmethod
    def _is_interactive():
        return not hasattr(__main__, '__file__')

    # display

    def _header(self, text):
        self._screen.header(text)

    def _subheader(self, text):
        self._screen.subheader(text)

    def _paragraph(self, *lines):
        self._screen.paragraph(*lines)

    def _plot(self, plot):
        self._screen.plot(plot)

    def _table(self, table):
        self._screen.table(table)

    @abstractmethod
    def _plot_target(self):
        pass

    @abstractmethod
    def _plot_category(self, name):
        pass

    @abstractmethod
    def _plot_number(self, name):
        pass

    @abstractmethod
    def _model():
        pass

    @abstractmethod
    def _benchmark_results(results, dev_y, rows):
        pass

    @abstractmethod
    def _score(act, pred):
        pass
