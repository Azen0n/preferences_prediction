import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from prettytable import PrettyTable
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from algorithms.metrics import chebyshev_distance, cosine_distance, euclidean_distance, \
    manhattan_distance, custom_accuracy
from preprocessing.preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS
from algorithms.decision_tree_classifier import test_sample_total_error, \
    test_sample_error_by_index


def calculating_distances(clf, music_test, movie_test):
    dist = [chebyshev_distance, cosine_distance, euclidean_distance, manhattan_distance]
    tab = PrettyTable()
    tab.field_names = ['distance', 'values']
    for distance in dist:
        error = test_sample_total_error(clf, music_test, movie_test, distance)
        tab.add_row([distance.__name__, error])
    print(tab)


def get_classifier_knn(music_genres: pd.DataFrame,
                                 movie_genres: pd.DataFrame) -> MultiOutputClassifier:
    x_train, x_test, y_train, y_test = train_test_split(music_genres, movie_genres, random_state=0)
    return MultiOutputClassifier(KNeighborsClassifier(n_neighbors=125),n_jobs=1).fit(x_train, y_train)


def main():
    music_genres = pd.read_csv('../data/responses.csv', usecols=MUSIC_GENRES_COLUMNS).fillna(value=3.0)
    movie_genres = pd.read_csv('../data/responses.csv', usecols=MOVIE_GENRES_COLUMNS).fillna(value=3.0)
    music_genres_train, music_genres_test, movie_genres_train, movie_genres_test = train_test_split(music_genres,
                                                                                                    movie_genres,
                                                                                                    random_state=0)
    clf = get_classifier_knn(music_genres, movie_genres)
    su = 0
    for i in range(100):
        custom_error = test_sample_error_by_index(clf, music_genres_test, movie_genres_test, custom_accuracy, i)
        su+=custom_error
        print(f'custom_error = {custom_error}')

    print(su/100)
    calculating_distances(clf,music_genres_test,movie_genres_test)

if __name__ == '__main__':
    main()
