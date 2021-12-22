import numpy as np
import pandas as pd
from preprocessing.preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS, get_genres
from algorithms.decision_tree_classifier import get_classifier_dtc


def preferences_input(genres: pd.DataFrame) -> np.ndarray:
    print('Enter number from 1 to 5 as your attitude towards each genre.')
    print('[Don\'t enjoy at all] 1 - 2 - 3 - 4 - 5 [Enjoy very much]')
    print('Count "3" as neutral/no attitude.')
    preferences = []
    for genre in genres:
        print(f'{genre} = ', end='')
        preferences.append(float(input()))
    return np.array(preferences)


def preferences_output(genres: pd.DataFrame, preferences: pd.DataFrame):
    perfect = 'We really recommend you: '
    nice = 'Perhaps you would like: '
    avoid = 'And probably you should avoid: '

    for genre in genres:
        if preferences[genre][0] == 5.0:
            perfect += f'{genre}, '
        elif preferences[genre][0] == 4.0:
            nice += f'{genre}, '
        elif preferences[genre][0] == 1.0:
            avoid += f'{genre}, '

    if perfect != 'We really recommend you: ':
        print(perfect[:-2])
    if nice != 'Perhaps you would like: ':
        print(nice[:-2])
    if avoid != 'Probably you should avoid: ':
        print(avoid[:-2])


def main():
    music_genres, movie_genres = get_genres()
    preferences = preferences_input(MUSIC_GENRES_COLUMNS)

    kwargs = {
        'criterion': 'gini',
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 3,
        'max_features': 'auto'
    }

    clf = get_classifier_dtc(music_genres, movie_genres, kwargs)
    prediction = clf.predict(pd.DataFrame([preferences], columns=MUSIC_GENRES_COLUMNS))
    prediction = pd.DataFrame(prediction, columns=MOVIE_GENRES_COLUMNS)
    preferences_output(MOVIE_GENRES_COLUMNS, prediction)


if __name__ == '__main__':
    main()
