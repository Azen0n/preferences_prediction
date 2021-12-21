import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt, cm
from preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS


def main():
    music_genres = pd.read_csv('../data/music_genres.csv')
    movie_genres = pd.read_csv('../data/movie_genres.csv')

    music_movies = pd.concat([music_genres, movie_genres], axis=1, keys=['music_genres', 'movie_genres'])
    correlation_matrix = music_movies.corr(method='spearman').loc['movie_genres', 'music_genres']
    m = music_genres.shape[1]
    n = movie_genres.shape[1]
    p_array = []
    x = 0
    for i, movie in enumerate(movie_genres):
        p_array.append([])
        for j, genre in enumerate(music_genres):
            pvalue = scipy.stats.kendalltau(music_genres[genre], movie_genres[movie])
            if pvalue[1] <= 0.05:
                x = x + 1
                p_array[i].append(1)
                # print('*', end=' ')
            else:
                # print('#', end=' ')
                p_array[i].append(0)
                x = x - 1

    plt.pcolor(p_array, cmap=cm.summer, edgecolors='k', linewidths=0.2)
    plt.xticks(range(len(MUSIC_GENRES_COLUMNS)), MUSIC_GENRES_COLUMNS, rotation=90,  ha='left')
    plt.yticks(range(len(MOVIE_GENRES_COLUMNS)), MOVIE_GENRES_COLUMNS, va="bottom")
    plt.subplots_adjust(bottom=0.3, left=0.2)
    plt.show()

    plt.matshow(correlation_matrix)
    plt.colorbar()
    plt.xticks(range(len(MUSIC_GENRES_COLUMNS)), MUSIC_GENRES_COLUMNS, rotation=45, ha="left")
    plt.yticks(range(len(MOVIE_GENRES_COLUMNS)), MOVIE_GENRES_COLUMNS)
    plt.show()


if __name__ == '__main__':
    main()
