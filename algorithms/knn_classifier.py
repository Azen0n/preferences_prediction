import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from algorithms.metrics import euclidean_distance, cross_validation, custom_accuracy
from preprocessing.preprocessing import get_genres


def get_classifier_knn(x_train: pd.DataFrame,
                       y_train: pd.DataFrame,
                       kwargs: dict) -> MultiOutputClassifier:
    return MultiOutputClassifier(KNeighborsClassifier(**kwargs), n_jobs=1).fit(x_train, y_train)


def main():
    music_genres, movie_genres = get_genres()

    kwargs = {
        'n_neighbors': 75,
        'weights': 'uniform',
        'metric': euclidean_distance
    }

    n = 5
    errors = []
    for _ in range(n):
        error = cross_validation(get_classifier_knn, kwargs, music_genres, movie_genres, 5, custom_accuracy)
        errors.append(error)

    print(f'Cross-validation Error (mean of {n}) = {np.mean(errors)}')


if __name__ == '__main__':
    main()
