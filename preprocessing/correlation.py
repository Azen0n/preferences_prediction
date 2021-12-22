import pandas as pd
from matplotlib import pyplot as plt
from preprocessing import MUSIC_GENRES_COLUMNS, MOVIE_GENRES_COLUMNS, get_genres


def main():
    music_genres, movie_genres = get_genres()

    music_movies = pd.concat([music_genres, movie_genres], axis=1, keys=['music_genres', 'movie_genres'])
    correlation_matrix = music_movies.corr(method='spearman').loc['movie_genres', 'music_genres']

    plt.matshow(correlation_matrix)
    plt.colorbar()
    plt.xticks(range(len(MUSIC_GENRES_COLUMNS)), MUSIC_GENRES_COLUMNS, rotation=45, ha="left")
    plt.yticks(range(len(MOVIE_GENRES_COLUMNS)), MOVIE_GENRES_COLUMNS)
    plt.show()


if __name__ == '__main__':
    main()
