import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS


def euclidean_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return np.linalg.norm(row1 - row2)


def manhattan_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return (np.absolute(row1 - row2)).sum()


def test_sample_error_by_index(clf: DecisionTreeClassifier,
                               x_test: pd.DataFrame,
                               y_test: pd.DataFrame,
                               distance,
                               index: int = None) -> float:
    """Prediction error of one test object by row index."""
    if index is None:
        index = np.random.randint(0, len(x_test))
    prediction = clf.predict(x_test.iloc[[index]])[0]
    real = y_test.iloc[[index]].values[0]
    return distance(prediction, real)


def test_sample_total_error(clf: DecisionTreeClassifier,
                            x_test: pd.DataFrame,
                            y_test: pd.DataFrame,
                            distance) -> float:
    """Mean prediction error of whole test subset."""
    y_predicted = clf.predict(x_test)
    errors = []
    for i in range(len(x_test)):
        errors.append(distance(y_predicted[i], y_test.iloc[[i]].values[0]))
    return np.mean(errors)


def get_classifier(music_genres: pd.DataFrame, movie_genres: pd.DataFrame) -> DecisionTreeClassifier:
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


if __name__ == '__main__':
    main()
