from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from src.models.recommender import RecommenderSystem


class MatrixFactorization(nn.Module, RecommenderSystem):

    def __init__(
        self,
        interactions: np.ndarray,
        n_factors: int,
        user_encoder: LabelEncoder,
        movie_encoder: LabelEncoder
    ):
        super(MatrixFactorization, self).__init__()

        self.DEVICE = torch.device("cpu")

        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")

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

    def forward(
        self,
        users: torch.LongTensor,
        movies: torch.LongTensor
    ) -> torch.FloatTensor:

        rating = self.user_embedding(users) * self.movie_embedding(movies)
        rating = rating.sum(1, keepdim=True)
        rating += self.user_bias(users)
        rating += self.movie_bias(movies)
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
        return movies

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

        user_id = torch.LongTensor([user_id]).to(self.DEVICE)
        movie_id = torch.LongTensor([movie_id]).to(self.DEVICE)

        rating = self.forward(user_id, movie_id)
        rating = rating.cpu().item()

        return rating

    def predict_scores(self, user_id: int) -> List[Tuple[int, float]]:
        """
        Predicts scores for all the movies, that a user would give.

        Parameters
        ----------
        user_id : int
            User's id from the data set.

        Returns
        -------
        float
            Ranked movies with their scores.
        """

        user_id = self.user_encoder.transform([user_id])[0]

        movies = set(range(self.MOVIE_DIM))
        movies_seen = self.interactions[user_id].nonzero()[0]

        movies -= set(movies_seen)
        movies = list(movies)
        movies = torch.LongTensor(movies).to(self.DEVICE)

        user_id = torch.LongTensor([user_id]).to(self.DEVICE)

        rating = self.forward(user_id, movies)
        rating, movies = torch.sort(rating, descending=True)

        rating = list(rating.cpu().numpy())
        movies = list(movies.cpu().numpy())

        movies = self.movie_encoder.inverse_transform(movies)

        return movies, rating
