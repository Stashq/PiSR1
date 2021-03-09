import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


def get_interactions(
    ratings: pd.DataFrame,
    user_encoder: LabelEncoder,
    movie_encoder: LabelEncoder
) -> np.ndarray:
    """
    Creates interaction matrix from ratings DataFrame.

    Parameters
    ----------
    ratings : pd.DataFrame
        Ratings DataFrame `[userId, movieId, rating, timestamp]`.
    user_encoder : LabelEncoder
        Encoder used to encode users' ids.
    movie_encoder : LabelEncoder
        Encoder used to encode movies' ids.

    Returns
    -------
    np.ndarray
        Interactions matrix. Rows are users, columns are movies.
        Specific cell denotes the rating, how user scored the movie.
        Interactions are encoded to handle continuity of indices.
    """

    users_encoded = user_encoder.transform(ratings['userId'].values)
    movies_encoded = movie_encoder.transform(ratings['movieId'].values)
    scores = ratings['rating']

    user_dim = len(user_encoder.classes_)
    movie_dim = len(movie_encoder.classes_)
    # interactions = sp.sparse.csr_matrix((user_dim, movie_dim), dtype=float)

    interactions = np.zeros((user_dim, movie_dim), dtype=float)

    iterator = tqdm(
        zip(users_encoded, movies_encoded, scores),
        desc='Building interaction matrix',
        total=len(users_encoded)
    )

    for user_id, movie_id, score in iterator:
        interactions[user_id, movie_id] = score

    return interactions


def get_sparsity_factor(array: np.ndarray) -> float:
    rows, _ = array.nonzero()
    sparsity_factor = len(rows) / array.size

    return sparsity_factor
