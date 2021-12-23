import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from algorithms.metrics import cross_validation, custom_accuracy
from preprocessing.preprocessing import get_genres


def get_classifier_dtc(x_train: pd.DataFrame,
                       y_train: pd.DataFrame,
                       kwargs: dict) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=0, **kwargs).fit(x_train, y_train)


def main():
    music_genres, movie_genres = get_genres()

    kwargs = {
        'criterion': 'gini',
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 3,
        'max_features': 'auto'
    }

    n = 100
    errors = []
    for _ in range(n):
        error = cross_validation(get_classifier_dtc, kwargs, music_genres, movie_genres, 5, custom_accuracy)
        errors.append(error)

    print(f'Cross-validation Error (mean of {n}) = {np.mean(errors)}')


if __name__ == '__main__':
    main()
