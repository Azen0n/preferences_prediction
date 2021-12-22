import itertools
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from algorithms.metrics import euclidean_distance, test_sample_total_error
from preprocessing.preprocessing import get_genres


def total_error(x_train: pd.DataFrame,
                x_test: pd.DataFrame,
                y_train: pd.DataFrame,
                y_test: pd.DataFrame,
                classifier: Callable,
                kwargs: dict) -> float:
    clf = classifier(**kwargs)
    clf.fit(x_train, y_train)
    error = test_sample_total_error(clf, x_test, y_test, euclidean_distance)
    return error


def get_parameters_list(tuned_parameters: dict) -> list:
    """Returns list of all combinations of parameters."""
    parameters_for_product = []
    # List of all parameters
    for i, parameter in enumerate(tuned_parameters):
        parameters_for_product.append([])
        for value in tuned_parameters[parameter]:
            parameters_for_product[i].append((parameter, value))

    parameters = []
    # Kind of cartesian product
    for i, combination in enumerate(itertools.product(*parameters_for_product)):
        parameters.append({})
        for parameter in combination:
            parameters[i][parameter[0]] = parameter[1]

    return parameters


def best_parameters_search(x: pd.DataFrame,
                           y: pd.DataFrame,
                           classifier: Callable,
                           tuned_parameters: dict) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    parameters = get_parameters_list(tuned_parameters)
    best_parameters = (1000.0, {})
    for params in parameters:
        error = total_error(x_train, x_test, y_train, y_test, classifier, params)
        if error < best_parameters[0]:
            best_parameters = (error, params)

    return best_parameters[1]


def main():
    dtc_tuned_parameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [i for i in range(4, 21)],
        'min_samples_split': [i for i in range(2, 7)],
        'min_samples_leaf': [i for i in range(1, 7)],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    }

    rfc_tuned_parameters = {
        'n_estimators': [i for i in range(135, 181, 5)],  # Number of trees
        'criterion': ['gini', 'entropy'],
        'max_depth': [i for i in range(10, 21)],
        'min_samples_split': [i for i in range(2, 5)],
        'min_samples_leaf': [i for i in range(1, 5)]
    }

    knn_tuned_parameters = {
        'n_neighbors': [i for i in range(21, 202, 2)],
        'weights': ['uniform'],
        'leaf_size': [i for i in range(3, 20)],
        'metric': [euclidean_distance]
    }

    music_genres, movie_genres = get_genres()
    dtc_best_parameters = best_parameters_search(music_genres,
                                                 movie_genres,
                                                 DecisionTreeClassifier,
                                                 dtc_tuned_parameters)

    print(f'Best parameters of Decision Tree Classifier: {dtc_best_parameters}')

    rfc_best_parameters = best_parameters_search(music_genres,
                                                 movie_genres,
                                                 RandomForestClassifier,
                                                 rfc_tuned_parameters)

    print(f'Best parameters of Random Forest Classifier: {rfc_best_parameters}')

    knn_best_parameters = best_parameters_search(music_genres,
                                                 movie_genres,
                                                 KNeighborsClassifier,
                                                 knn_tuned_parameters)

    print(f'Best parameters of K-Nearest Neighbors: {knn_best_parameters}')


if __name__ == '__main__':
    main()
