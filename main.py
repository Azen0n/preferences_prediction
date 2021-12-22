import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split

from algorithms.decision_tree_classifier import get_classifier_dtc
from algorithms.knn_classifier import get_classifier_knn
from algorithms.random_forest_classifier import get_classifier_rfc
from algorithms.metrics import euclidean_distance
from preferences_input_output import preferences_input, preferences_output
from preprocessing.preprocessing import get_genres, MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS


def get_result(clf: ClassifierMixin, preferences: np.ndarray):
    prediction = clf.predict(pd.DataFrame([preferences], columns=MUSIC_GENRES_COLUMNS))
    prediction = pd.DataFrame(prediction, columns=MOVIE_GENRES_COLUMNS)
    preferences_output(MOVIE_GENRES_COLUMNS, prediction)


def main():
    music_genres, movie_genres = get_genres()

    music_genres_train, music_genres_test, movie_genres_train, movie_genres_test = train_test_split(music_genres,
                                                                                                    movie_genres,
                                                                                                    random_state=0)
    kwargs = {
        'dtc': {
            'criterion': 'gini',
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 3,
            'max_features': 'auto'
        },
        'rfc': {
            'n_estimators': 170,
            'criterion': 'entropy',
            'max_depth': 12,
            'min_samples_split': 3,
            'min_samples_leaf': 1
        },
        'knn': {
            'n_neighbors': 75,
            'weights': 'uniform',
            'metric': euclidean_distance
        }
    }

    dtc_clf = get_classifier_dtc(music_genres_train, movie_genres_train, kwargs['dtc'])
    rfc_clf = get_classifier_rfc(music_genres_train, movie_genres_train, kwargs['rfc'])
    knn_clf = get_classifier_knn(music_genres_train, movie_genres_train, kwargs['knn'])

    preferences = preferences_input(MUSIC_GENRES_COLUMNS)

    print('\nDecision Tree:')
    get_result(dtc_clf, preferences)
    print('\nRandom Forest:')
    get_result(rfc_clf, preferences)
    print('\nK Nearest Neighbors:')
    get_result(knn_clf, preferences)


if __name__ == '__main__':
    main()
