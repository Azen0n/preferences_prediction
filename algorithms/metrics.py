from typing import Callable
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.base import ClassifierMixin


def euclidean_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return np.linalg.norm(row1 - row2)


def manhattan_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return (np.absolute(row1 - row2)).sum()


def chebyshev_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return (np.absolute(row1 - row2)).max()


def cosine_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return np.dot(row1, row2) / (np.sqrt((row1 ** 2).sum()) + np.sqrt((row2 ** 2).sum()))


def custom_accuracy(row1: np.ndarray, row2: np.ndarray) -> float:
    error = []
    for i in range(row1.shape[0]):
        error.append(100.0 - (np.absolute(row2[i] - row1[i]) * 25.0))
    return float(np.mean(error))


def test_sample_error_by_index(clf: ClassifierMixin,
                               x_test: pd.DataFrame,
                               y_test: pd.DataFrame,
                               distance: Callable[[np.ndarray, np.ndarray], float],
                               index: int = None) -> float:
    """Prediction error of one test object by row index."""
    if index is None:
        index = np.random.randint(0, len(x_test))
    prediction = clf.predict(x_test.iloc[[index]])[0]
    real = y_test.iloc[[index]].values[0]
    return distance(prediction, real)


def test_sample_total_error(clf: ClassifierMixin,
                            x_test: pd.DataFrame,
                            y_test: pd.DataFrame,
                            distance: Callable[[np.ndarray, np.ndarray], float]) -> float:
    """Mean prediction error of whole test subset."""
    y_predicted = clf.predict(x_test)
    errors = []
    for i in range(len(x_test)):
        errors.append(distance(y_predicted[i], y_test.iloc[[i]].values[0]))
    return float(np.mean(errors))


def total_error_for_each_distance(clf: ClassifierMixin,
                                  music_test: pd.DataFrame,
                                  movie_test: pd.DataFrame):
    distances = [chebyshev_distance, cosine_distance, euclidean_distance, manhattan_distance]
    table = PrettyTable()
    table.field_names = ['distance', 'values']
    for distance in distances:
        error = test_sample_total_error(clf, music_test, movie_test, distance)
        table.add_row([distance.__name__, error])
    print(table)
