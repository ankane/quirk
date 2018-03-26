from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import __main__

# analyze
import numpy as np
import pandas as pd
import seaborn as sns

# model
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer

try:
    import Geohash
except ImportError:
    pass

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
                 id_col=None, datetime_cols=None, eval_metric=None):

        train_df = self._fetch_data(train_data)
        test_df = self._fetch_data(test_data)

        if test_df is not None:
            self._id_series = test_df[id_col]

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
        self._eval_metric = eval_metric
        self._seed = 2016

        self._train_features_df = None
        self._test_features_df = None

        self.column_types = {}

        if self._is_interactive():
            self._screen = Notebook()
        else:
            self._screen = Terminal()

    def analyze(self):
        self._summarize_data()
        self._show_target_variable()
        self._show_pairwise_correlation()
        self._show_features()

    def analyze_column(self, name):
        self._generate_features(cols=[name])

    def model(self):
        if 'xgb' not in globals():
            # TODO better error type
            raise ValueError('XGBoost not installed')

        self._header('Model')

        target_col = self._target_col
        id_col = self._id_col

        if self._train_features_df is None:
            self._generate_features(viz=False)

        if len(self._train_features_df.columns) < 2:
            # TODO better error type
            raise ValueError('No features')

        train_model_df = self._train_features_df.drop(target_col, axis=1)
        test_model_df = self._test_features_df.copy()
        y = self._y()

        # dummy vars
        for f in train_model_df:
            col = train_model_df[f]

            if col.dtype == 'object':
                default_value = self._mode(col)

                # train_col = col.fillna(default_value)
                # test_col = [] if self._test_df is None else self._test_df[col.name].fillna(default_value)

                # categories = np.union1d(train_col, test_col)
                # train_dummies = pd.get_dummies(train_col.astype('category', categories=categories), prefix=col.name)
                # train_model_df = pd.concat([train_model_df.drop(col.name, axis=1), train_dummies], axis=1)

                # if self._test_df is not None:
                #     test_dummies = pd.get_dummies(test_col.astype('category', categories=categories), prefix=col.name)
                #     test_model_df = pd.concat([test_model_df.drop(col.name, axis=1), test_dummies], axis=1)

                le = preprocessing.LabelEncoder()
                train_model_df[f] = le.fit_transform(train_model_df[f].fillna(default_value))
                if self._test_df is not None:
                    test_model_df[f] = le.fit_transform(test_model_df[f].fillna(default_value))

        if self._eval_metric == 'mlogloss':
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)
            self._classes = le.classes_

        # tried stratify=y, but produced worse results
        dev_x, val_x, dev_y, val_y = model_selection.train_test_split(
            train_model_df, y, test_size=0.33, random_state=self._seed)

        # benchmark models
        predictions = self._build_models(dev_x, dev_y, val_x, val_y)
        self._predictions = predictions

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
        self._paragraph('Score: %f' % scores['XGBoost'])
        self._plot(xgb.plot_importance(self._xgboost_model, max_num_features=20))

        # save data
        if self._test_df is not None:
            self._save_results(train_model_df, y, test_model_df)

    def _save_results(self, train_x, train_y, test_x):
        self._header('Test Predictions')

        # retrain model on all data
        preds = self._xgboost_predict(train_x, train_y, test_x, None)

        out_df = pd.DataFrame()
        out_df[self._id_col] = self._id_series.values

        if self._eval_metric == 'mlogloss':
            for i, label in enumerate(self._classes):
                out_df[label] = preds[:, i]
        else:
            out_df[self._target_col] = preds

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

        lines = ['Training set size: %d' % len(self._train_df)]
        if self._test_df is not None:
            lines.append('Test set size: %d' % len(self._test_df))

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
    def _generate_features(self, viz=True, cols=None):
        target_col = self._target_col

        self._train_features_df = pd.DataFrame()
        self._test_features_df = pd.DataFrame()
        self._train_features_df[target_col] = self._train_df[target_col]

        for name, col in self._train_df.iteritems():
            if name == target_col:
                continue

            if cols != None and not name in cols:
                continue

            info = self._col_info(col)
            unique_count = info['unique_count']
            null_count = info['null_count']
            column_type = info['column_type']

            self.column_types[name] = column_type

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
                self._train_features_df[col.name + '_week'] = col.apply(lambda x: x.isocalendar()[1])
                self._train_features_df[col.name + '_month'] = col.dt.month
                self._train_features_df[col.name + '_year'] = col.dt.year
                if self._test_df is not None:
                    self._test_features_df[col.name + '_hour'] = self._test_df[col.name].dt.hour
                    self._test_features_df[col.name + '_weekday'] = self._test_df[col.name].dt.weekday
                    self._test_features_df[col.name + '_week'] = self._test_df[col.name].apply(lambda x: x.isocalendar()[1])
                    self._test_features_df[col.name + '_month'] = self._test_df[col.name].dt.month
                    self._test_features_df[col.name + '_year'] = self._test_df[col.name].dt.year

            elif column_type == 'text':
                self._train_features_df[col.name + '_num_words'] = self._word_count(col)
                self._test_features_df[col.name + '_num_words'] = self._word_count(col)

                # might take a while
                self._paragraph('Processing ' + col.name + '...')

                self._bag_of_words(col, max_features=100)
            elif column_type == 'list':
                self._train_features_df[col.name + '_count'] = col.apply(len)
                if self._test_df is not None:
                    self._test_features_df[col.name + '_count'] = self._test_df[col.name].apply(len)

                self._bag_of_words(col, max_features=20, transform=lambda x: " ".join(x))

            # visualize
            if viz:
                if column_type == 'category' and unique_count <= 20:
                    self._plot_category(col.name)
                elif column_type == 'number':
                    self._plot_number(col.name)
                elif column_type == 'time':
                    self._plot_category(col.name + '_hour')
                    self._plot_category(col.name + '_weekday')
                    self._plot_category(col.name + '_week')
                    self._plot_category(col.name + '_month')
                    self._plot_category(col.name + '_year')

        if cols is None:
            self._process_geo(viz=viz)

    def _bag_of_words(self, col, max_features=100, transform=None):
        vectorizer = CountVectorizer(analyzer='word',
                                     stop_words='english',
                                     max_features=max_features)

        train_col = col
        if transform is not None:
            train_col = train_col.apply(transform)

        train_features = vectorizer.fit_transform(train_col)
        cols2 = [col.name + '_word_' + x for x in vectorizer.get_feature_names()]
        arr = pd.DataFrame(train_features.toarray(), columns=cols2).set_index(self._train_features_df.index)
        self._train_features_df = pd.concat([self._train_features_df, arr], axis=1)

        if self._test_df is not None:
            test_col = self._test_df[col.name]
            if transform is not None:
                test_col = test_col.apply(transform)

            test_features = vectorizer.transform(test_col)
            arr = pd.DataFrame(test_features.toarray(), columns=cols2).set_index(self._test_features_df.index)
            self._test_features_df = pd.concat([self._test_features_df, arr], axis=1)

    @staticmethod
    def _show_pct(message, num, denom):
        message = message % num
        if num > 0:
            pct = 100.0 * num / denom
            message += ' (%.1f%%)' % round(pct, 1)
        return message

    def _process_geo(self, viz=True):
        train_df = self._train_df
        test_df = self._test_df

        geo_cols = ['latitude', 'longitude']

        # latitude and longtitude
        if all(x in train_df.keys() for x in geo_cols):
            if viz:
                self._subheader("latitude & longitude")

                # drop outliers
                data = train_df.dropna(subset=geo_cols)
                for col in geo_cols:
                    data = self._drop_outliers(data, col)

                # TODO use map
                self._plot(sns.lmplot(y='latitude', x='longitude', hue='interest_level', fit_reg=False, data=data))

            if 'Geohash' in globals():
                # TODO handle nulls
                # precision = 8
                # self._train_features_df['geohash'] = train_df.apply(lambda x: Geohash.encode(x['latitude'], x['longitude'], precision=precision), axis=1)
                # if self._test_df is not None:
                #     self._test_features_df['geohash'] = test_df.apply(lambda x: Geohash.encode(x['latitude'], x['longitude'], precision=precision), axis=1)
                pass
            else:
                self._paragraph("Install the Geohash package for better feature engineering")

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
    def _word_count(col):
        return col.apply(lambda x: len((x or '').split(' ')))

    def _col_info(self, col):
        total_count = len(col)

        average_words = None
        try:
            unique_count = len(col.unique())
            if col.dtype == 'object':
                try:
                    word_count = self._word_count(col)
                    average_words = word_count.where(word_count > 0).mean()
                except AttributeError:
                    pass

        except TypeError:
            unique_count = len(col.apply(tuple).unique())


        # if col.dtype == 'object':
        #     # TODO distinguish betewen blank and missing
        #     null_count = col.apply(lambda x: len(x or '') == 0).sum()
        # else:
        null_count = col.isnull().sum()

        numeric = col.dtype == 'int64' or col.dtype == 'float64'
        if col.dtype == 'datetime64[ns]':
            column_type = 'time'
        elif col.dtype == 'object' and type(col.values[0]) is list:
            column_type = 'list'
        elif average_words != None and average_words > 10:
            column_type = 'text'
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

    @staticmethod
    def _mode(col):
        return col.value_counts().index[0]

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
    def _xgboost_predict(self, train_x, train_y, test_x, test_y):
        pass

    @abstractmethod
    def _build_models(self, train_x, train_y, test_x, test_y):
        pass

    @abstractmethod
    def _score(self, act, pred):
        pass
