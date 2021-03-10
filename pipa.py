# from src.models.hybrid import Hybrid
from typing import Set
import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy.sparse import coo_matrix, csr_matrix

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.util.data import get_interactions, get_train_test_ratings

MOVIES = "data/movies_metadata.csv"


def getMoviesContentDataset(
    movie_encoder: LabelEncoder,
    movies_to_keep: Set[int]
):
    movies = pd.read_csv(MOVIES)
    to_drop = ["1997-08-20", "2012-09-29", "2014-01-01"]
    for drop_error in to_drop:
        movies = movies[movies.id != drop_error]

    movies.id = movies.id.astype('int64')
    movies = movies.loc[movies['id'].isin(movies_to_keep)]
    movies.id = movie_encoder.transform(movies.id)
    movies_to_keep = movie_encoder.transform(list(movies_to_keep))

    movies = movies.drop_duplicates(subset=['id'], keep='first')

    budget_Scaler = StandardScaler().fit(movies.budget.to_numpy().reshape(-1, 1))

    lang_encoder = LabelEncoder().fit(movies.original_language)

    genres_encoder = LabelEncoder().fit(movies.genres)

    spoken_languages_encoder = LabelEncoder().fit(movies.spoken_languages)

    adult_encoder = LabelEncoder().fit(movies.adult)

    belongs_to_collection_encoder = LabelEncoder().fit(
        movies.belongs_to_collection
    )

    movies.belongs_to_collection = belongs_to_collection_encoder.transform(
        movies.belongs_to_collection
    )
    movies.original_language = lang_encoder.transform(
        movies.original_language
    )
    movies.adult = adult_encoder.transform(movies.adult)
    movies.spoken_languages = spoken_languages_encoder.transform(
        movies.spoken_languages
    )
    movies.genres = genres_encoder.transform(movies.genres)
    movies.budget = budget_Scaler.transform(
        movies.budget.to_numpy().reshape(-1, 1)
    )
    movies.popularity = movies.popularity.astype("float64")

    movies = movies.drop(["homepage", "imdb_id", "original_title",
                          "overview", "poster_path",
                          "production_companies", "production_countries",
                          "release_date", "status", "tagline",
                          "title", "video"], axis=1)
    movies = movies.set_index("id")
    movies = movies.dropna()

    not_included_ids = [idx for idx in movies_to_keep
                        if idx not in movies.index]
    missing_movies = [[0] * movies.shape[1]] * len(not_included_ids)
    missing_movies = pd.DataFrame(
        data=missing_movies,
        columns=movies.columns,
        index=not_included_ids
    )
    movies = pd.concat([movies, missing_movies])
    movies = movies.sort_index()
    return movies


def main():
    RATINGS_PATH = Path('data/ratings_small.csv')
    ratings = pd.read_csv(RATINGS_PATH)

    user_encoder = LabelEncoder()
    user_encoder.fit(ratings['userId'].values)

    movie_encoder = LabelEncoder()
    movie_encoder.fit(ratings['movieId'].values)

    train_ratings, test_ratings = get_train_test_ratings(ratings)

    train_interactions = get_interactions(
        train_ratings,
        user_encoder,
        movie_encoder
    )
    train_interactions = coo_matrix(train_interactions)

    test_interactions = get_interactions(
        test_ratings,
        user_encoder,
        movie_encoder
    )
    test_interactions = coo_matrix(test_interactions)

    movies = getMoviesContentDataset(
        movie_encoder,
        movies_to_keep=set(ratings["movieId"].values)
    )
    movies = csr_matrix(movies.values)
    model = LightFM()
    model.fit(
        interactions=train_interactions,
        item_features=movies,
        epochs=5,
        num_threads=2
    )

if __name__ == '__main__':
    main()
