# Quirk

Build powerful predictive models with a few lines of code

[See a demo](https://github.com/ankane/quirk/blob/demos/demos/Titanic.ipynb)

Quirk does:

- exploratory data analysis
- feature engineering
- predictive modeling

## Installation

With pip, run:

```sh
pip install quirk
```

## Getting Started

For rich visualizations, run Quirk from a [Jupyter notebook](https://jupyter.org/).

For classification, use:

```python
import quirk

qk = quirk.Classifier(
    train_data='train.csv',
    test_data='test.csv',
    target_col='Survived',
    id_col='PassengerId')

qk.analyze()
qk.model()
```

For regression, use the `quirk.Regressor` class.

**Tip:** To prevent scrolling in notebooks, select `Cell > Current Outputs > Toggle Scrolling`.

## Features

There are two primary methods:

- `analyze` runs exploratory data analysis
- `model` builds and evaluates different models

Optionally pass test data if you want to generate a CSV file with predictions.

## Data

Data can be a file

```python
quirk.Classifier(train_data='train.csv', ...)
```

Or a data frame

```python
train_df = pd.read_csv('train.csv')

# do preprocessing
# ...

quirk.Classifier(train_data=train_df, ...)
```

Specify datetime columns with:

```python
quirk.Classifier(datetime_cols=['created'], ...)
```

## Evaluation

Quirk has support for a number of eval metrics.

Classification

- `accuracy` - # correct / total (default)
- `auc` - area under the ROC curve
- `mlogloss` - multi class log loss

Regression

- `rmse` - root mean square error (default)
- `rmsle` - root mean square logarithmic error

Specify an eval metric with:

```python
quirk.Classifier(eval_metric='mlogloss', ...)
```

## Modeling

Quirk builds and compares different models. Currently, it uses:

1. boosted trees
2. simple benchmarks (mode for classification, mean and median for regression)

[XGBoost](https://github.com/dmlc/xgboost) is required for boosted trees. See [how to install](https://xgboost.readthedocs.io/en/latest/build.html). On Mac, use:

```sh
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; cp make/minimum.mk ./config.mk; make -j4
cd python-package; sudo python setup.py install
```

## Performance

Dataset | Eval Metric | v0.1 | Current
--- | --- | --- | ---
[Titanic](https://www.kaggle.com/c/titanic) | Accuracy | 0.77512 | 0.77512
[Rental Listing Inquiries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries) | Multi Class Log Loss | - | 0.61861
[House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | RMSLE | 0.14069 | 0.13355

## History

View the [changelog](https://github.com/ankane/quirk/blob/master/CHANGELOG.md)

## TODO

- Improve model performance
- Text features
- Geo features
- Name and address features
- Hyper-parameter tuning

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/quirk/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/quirk/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features
