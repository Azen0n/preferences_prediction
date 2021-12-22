import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from algorithms.metrics import custom_accuracy, total_error_for_each_distance
from preprocessing.preprocessing import get_genres
from algorithms.decision_tree_classifier import test_sample_error_by_index


def get_classifier_knn(music_genres: pd.DataFrame,
                       movie_genres: pd.DataFrame) -> MultiOutputClassifier:
    x_train, x_test, y_train, y_test = train_test_split(music_genres, movie_genres, random_state=0)
    return MultiOutputClassifier(KNeighborsClassifier(n_neighbors=125), n_jobs=1).fit(x_train, y_train)


def main():
    music_genres, movie_genres = get_genres()
    music_genres_train, music_genres_test, movie_genres_train, movie_genres_test = train_test_split(music_genres,
                                                                                                    movie_genres,
                                                                                                    random_state=0)
    clf = get_classifier_knn(music_genres, movie_genres)

    mean_error = []
    for i in range(100):
        custom_error = test_sample_error_by_index(clf, music_genres_test, movie_genres_test, custom_accuracy, i)
        mean_error.append(custom_error)
        print(f'custom_error = {custom_error}')

    print(f'mean_error = {np.mean(mean_error)}')
    total_error_for_each_distance(clf, music_genres_test, movie_genres_test)


if __name__ == '__main__':
    main()
