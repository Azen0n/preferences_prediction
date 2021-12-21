import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from algorithms.metrics import euclidean_distance, custom_accuracy, test_sample_error_by_index, test_sample_total_error
from preprocessing.preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS


def get_classifier(music_genres: pd.DataFrame,
                   movie_genres: pd.DataFrame) -> DecisionTreeClassifier:
    x_train, x_test, y_train, y_test = train_test_split(music_genres, movie_genres, random_state=0)
    return DecisionTreeClassifier(random_state=0).fit(x_train, y_train)


def main():
    music_genres = pd.read_csv('../data/responses.csv', usecols=MUSIC_GENRES_COLUMNS).fillna(value=3.0)
    movie_genres = pd.read_csv('../data/responses.csv', usecols=MOVIE_GENRES_COLUMNS).fillna(value=3.0)

    music_genres_train, music_genres_test, movie_genres_train, movie_genres_test = train_test_split(music_genres,
                                                                                                    movie_genres,
                                                                                                    random_state=0)
    clf = get_classifier(music_genres_train, movie_genres_train)

    error = test_sample_error_by_index(clf, music_genres_test, movie_genres_test, euclidean_distance, 0)
    print(f'error = {error}')
    total_error = test_sample_total_error(clf, music_genres_test, movie_genres_test, euclidean_distance)
    print(f'total_error = {total_error}')
    for i in range(100):
        custom_error = test_sample_error_by_index(clf, music_genres_test, movie_genres_test, custom_accuracy, i)
        print(f'custom_error = {custom_error}')


if __name__ == '__main__':
    main()
