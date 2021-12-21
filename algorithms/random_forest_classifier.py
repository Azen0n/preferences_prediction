import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from algorithms.metrics import custom_accuracy, test_sample_error_by_index, total_error_for_each_distance
from preprocessing.preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS


def get_classifier_random_forest(music_genres: pd.DataFrame,
                                 movie_genres: pd.DataFrame) -> RandomForestClassifier:
    x_train, x_test, y_train, y_test = train_test_split(music_genres, movie_genres, random_state=0)
    return RandomForestClassifier(random_state=0).fit(x_train, y_train)


def main():
    music_genres = pd.read_csv('../data/responses.csv', usecols=MUSIC_GENRES_COLUMNS).fillna(value=3.0)
    movie_genres = pd.read_csv('../data/responses.csv', usecols=MOVIE_GENRES_COLUMNS).fillna(value=3.0)
    music_genres_train, music_genres_test, movie_genres_train, movie_genres_test = train_test_split(music_genres,
                                                                                                    movie_genres,
                                                                                                    random_state=0)
    clf = get_classifier_random_forest(music_genres, movie_genres)
    for i in range(100):
        custom_error = test_sample_error_by_index(clf, music_genres_test, movie_genres_test, custom_accuracy, i)
        print(f'custom_error = {custom_error}')
    total_error_for_each_distance(clf, music_genres_test, movie_genres_test)


if __name__ == '__main__':
    main()
