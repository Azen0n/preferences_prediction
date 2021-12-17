import numpy as np
import pandas as pd
from preprocessing.preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS
from algorithms.decision_tree_classifier import get_classifier


def preferences_input(genres: pd.DataFrame) -> np.ndarray:
    print('Enter number from 1 to 5 as your attitude towards each genre.')
    print('[Don\'t enjoy at all] 1 - 2 - 3 - 4 - 5 [Enjoy very much]')
    print('Count "3" as neutral/no attitude.')
    preferences = []
    for genre in genres:
        print(f'{genre} = ', end='')
        preferences.append(float(input()))  # TODO: Data validation?
    return np.array(preferences)


def main():
    music_genres = pd.read_csv('data/responses.csv', usecols=MUSIC_GENRES_COLUMNS).fillna(value=3.0)
    movie_genres = pd.read_csv('data/responses.csv', usecols=MOVIE_GENRES_COLUMNS).fillna(value=3.0)
    preferences = preferences_input(MUSIC_GENRES_COLUMNS)
    clf = get_classifier(music_genres, movie_genres)
    prediction = clf.predict(pd.DataFrame([preferences], columns=MUSIC_GENRES_COLUMNS))
    print(f'prediction = {prediction}')


if __name__ == '__main__':
    main()
