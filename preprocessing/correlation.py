import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from matplotlib import pyplot as plt, colors as clrs
from preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS, get_genres


def correlation_matrix():
    music_genres, movie_genres = get_genres()

    music_movies = pd.concat([music_genres, movie_genres], axis=1, keys=['music_genres', 'movie_genres'])
    corr_matrix = music_movies.corr(method='spearman').loc['movie_genres', 'music_genres']

    plt.matshow(corr_matrix)
    plt.colorbar()
    plt.xticks(range(len(MUSIC_GENRES_COLUMNS)), MUSIC_GENRES_COLUMNS, rotation=45, ha='left')
    plt.yticks(range(len(MOVIE_GENRES_COLUMNS)), MOVIE_GENRES_COLUMNS)
    plt.show()


def p_value_matrix():
    music_genres, movie_genres = get_genres()

    p_value_array = np.zeros((movie_genres.shape[1], music_genres.shape[1]))
    for i, movie_genre in enumerate(movie_genres):
        for j, music_genre in enumerate(music_genres):
            p_value = kendalltau(music_genres[music_genre], movie_genres[movie_genre])[1]
            if p_value <= 0.05:
                p_value_array[i][j] = 1

    colors = clrs.LinearSegmentedColormap.from_list('', ['midnightblue', 'teal'])
    plt.pcolor(p_value_array, edgecolors='k', linewidths=0.2, cmap=colors)
    plt.xticks(range(len(MUSIC_GENRES_COLUMNS)), MUSIC_GENRES_COLUMNS, rotation=90, ha='left')
    plt.yticks(range(len(MOVIE_GENRES_COLUMNS)), MOVIE_GENRES_COLUMNS, va='bottom')
    plt.colorbar()
    plt.subplots_adjust(bottom=0.3, left=0.2)
    plt.show()


def main():
    correlation_matrix()
    p_value_matrix()


if __name__ == '__main__':
    main()
