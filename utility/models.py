import numpy as np
from typing import Any, Dict
from sklearn import linear_model
from sklearn.base import RegressorMixin


def get_regression_model(model_type: str,
                         **kwargs):
    if model_type == 'linear':
        model = linear_model.LinearRegression(**kwargs)
    elif model_type == 'logistic':
        model = linear_model.LogisticRegression(**kwargs)
    elif model_type == 'sgd-regressor':
        model = linear_model.SGDRegressor(**kwargs)
    elif model_type == 'lasso':
        model = linear_model.Lasso(**kwargs)
    else:
        raise ValueError(f"Error: unknown model_type = [{model_type}]")
    return model


def fit_regression_model(x: np.ndarray,
                         y: np.ndarray,
                         model_type: str = 'linear',
                         model_kwargs: Dict[str, Any] = None,
                         ):
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    if model_kwargs is None:
        model_kwargs = dict()

    if x.ndim == 1:
        x = x[..., np.newaxis]

    if y.ndim != 1:
        raise ValueError(f"Error: expects 1D target, found [{y.shape}]!")

    if x.ndim != 2:
        raise ValueError(f"Error: expects 2D feature vector, found [{x.shape}]!")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Error: input and output dimension does not match!")

    model = get_regression_model(model_type=model_type,
                                 **model_kwargs)
    model.fit(x, y)
    return model


def regressor_prediction(model,
                         x: np.ndarray,
                         ):
    x = np.array(x).astype(float)
    if x.ndim != 2:
        x = x[..., np.newaxis]
    return model.predict(x)

