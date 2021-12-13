import pandas as pd

MUSIC_GENRES_COLUMNS = ['Dance',
                        'Folk',
                        'Country',
                        'Classical music',
                        'Musical',
                        'Pop',
                        'Rock',
                        'Metal or Hardrock',
                        'Punk',
                        'Hiphop, Rap',
                        'Reggae, Ska',
                        'Swing, Jazz',
                        'Rock n roll',
                        'Alternative',
                        'Latino',
                        'Techno, Trance',
                        'Opera']

MOVIE_GENRES_COLUMNS = ['Horror',
                        'Thriller',
                        'Comedy',
                        'Romantic',
                        'Sci-fi',
                        'War',
                        'Fantasy/Fairy tales',
                        'Animated',
                        'Documentary',
                        'Western',
                        'Action']


def main():
    music_columns_general = ['Music',  # I enjoy listening to music.
                             'Slow songs or fast songs']

    movies_columns_general = ['Movies']  # I really enjoy watching movies.

    music_genres = pd.read_csv('data/responses.csv', usecols=MUSIC_GENRES_COLUMNS)
    movie_genres = pd.read_csv('data/responses.csv', usecols=MOVIE_GENRES_COLUMNS)

    music_genres.to_csv('data/music_genres.csv', index=False)
    movie_genres.to_csv('data/movie_genres.csv', index=False)


if __name__ == '__main__':
    main()
