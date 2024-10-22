from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from src.models.recommender import RecommenderSystem

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


class MatrixFactorization(nn.Module, RecommenderSystem):

    def __init__(
        self,
        interactions: np.ndarray,
        n_factors: int,
        user_encoder: LabelEncoder,
        movie_encoder: LabelEncoder
    ):
        super(MatrixFactorization, self).__init__()

        self.MAX_RATING = 5
        self.MIN_RATING = 0

        self.interactions = interactions
        self.USER_DIM, self.MOVIE_DIM = self.interactions.shape
        self.N_FACTORS = n_factors

        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder

        self.setup()

    def setup(self):
        self.user_embedding = nn.Embedding(
            self.USER_DIM,
            self.N_FACTORS,
            sparse=True
        )

        self.movie_embedding = nn.Embedding(
            self.MOVIE_DIM,
            self.N_FACTORS,
            sparse=True
        )

        self.user_bias = nn.Embedding(self.USER_DIM, 1, sparse=True)
        self.movie_bias = nn.Embedding(self.MOVIE_DIM, 1, sparse=True)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        users: torch.LongTensor,
        movies: torch.LongTensor
    ) -> torch.FloatTensor:

        rating = self.user_embedding(users) * self.movie_embedding(movies)
        rating = rating.sum(1, keepdim=True)
        rating += self.user_bias(users)
        rating += self.movie_bias(movies)
        rating = self.sigmoid(rating)
        rating = rating.squeeze()
        return rating

    def predict(self, user_id: int) -> List[int]:
        """
        Predicts ranking of movies to watch for a user.

        Parameters
        ----------
        user_id : int
            User's id from the data set.

        Returns
        -------
        List[int]
            List of movies ids. Best recommendations first.
        """
        movies, rating = self.predict_scores(user_id)
        return list(movies)

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """
        Predicts score for a given movie that a user would give.

        Parameters
        ----------
        user_id : int
            User's id from the data set.
        movie_id : int
            Movie's id from the data set.

        Returns
        -------
        float
            Predicted movie's score in range [0, 5].
        """
        user_id = self.user_encoder.transform([user_id])[0]
        movie_id = self.movie_encoder.transform([movie_id])[0]

        user_id = torch.LongTensor([user_id]).to(DEVICE)
        movie_id = torch.LongTensor([movie_id]).to(DEVICE)

        rating = self.forward(user_id, movie_id)
        rating = rating.cpu().item()

        return rating * self.MAX_RATING

    def predict_scores(self, user_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts scores for all the movies, that a user would give.

        Parameters
        ----------
        user_id : int
            User's id from the data set.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]:
            Ranked movies with their scores.
        """

        user_id = self.user_encoder.transform([user_id])[0]

        movies = set(range(self.MOVIE_DIM))
        movies_seen = self.interactions[user_id].nonzero()[0]

        movies -= set(movies_seen)
        movies = list(movies)
        movies = torch.LongTensor(movies).to(DEVICE)

        user = torch.LongTensor([user_id]).to(DEVICE)

        ratings = self.forward(user, movies) * self.MAX_RATING

        ratings = list(ratings.cpu().numpy())
        movies = list(movies.cpu().numpy())

        ranking = pd.DataFrame(
            zip(movies, ratings),
            columns=['movie', 'rating']
        )

        ranking = ranking.sort_values(
            by='rating',
            ascending=False
        )

        movies = ranking['movie'].values
        ratings = ranking['rating'].values

        movies = self.movie_encoder.inverse_transform(movies)

        return movies, ratings
