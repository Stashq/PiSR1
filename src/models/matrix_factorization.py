from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from src.models.recommender import RecommenderSystem
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

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

        user_id = torch.LongTensor([user_id]).to(DEVICE)
        movie_id = torch.LongTensor([movie_id]).to(DEVICE)

        rating = self.forward(user_id, movie_id)
        rating = rating.cpu().item()

        return rating

    def predict_scores(self, user_id: int) -> Tuple[List[int], List[float]]:
        """
        Predicts scores for all the movies, that a user would give.

        Parameters
        ----------
        user_id : int
            User's id from the data set.

        Returns
        -------
        Tuple[List[int], List[float]]
            Ranked movies with their scores.
        """

        user_id = self.user_encoder.transform([user_id])[0]

        movies = set(range(self.MOVIE_DIM))
        movies_seen = self.interactions[user_id].nonzero()[0]

        movies -= set(movies_seen)
        movies = list(movies)
        movies = torch.LongTensor(movies).to(DEVICE)

        user_id = torch.LongTensor([user_id]).to(DEVICE)

        rating = self.forward(user_id, movies)
        rating, movies = torch.sort(rating, descending=True)

        rating = list(rating.cpu().numpy())
        movies = list(movies.cpu().numpy())

        movies = self.movie_encoder.inverse_transform(movies)
        movies = list(movies)

        return movies, rating


def get_dataset(interactions: np.ndarray) -> TensorDataset:
    users, movies = interactions.nonzero()
    ratings = interactions[users, movies]

    permutation = torch.randperm(len(users))

    users = users[permutation]
    movies = movies[permutation]
    ratings = ratings[permutation]

    users = torch.LongTensor(users).to(DEVICE)
    movies = torch.LongTensor(movies).to(DEVICE)
    ratings = torch.FloatTensor(ratings).to(DEVICE)

    return TensorDataset(users, movies, ratings)


def train(
    model: MatrixFactorization,
    train_interactions: np.ndarray,
    test_interactions: np.ndarray,
    epochs: int,
    batch_size: int,
    verbose: int = 0
):
    loss_func = nn.MSELoss()

    optimizer = optim.SparseAdam(model.parameters(), lr=1e-3)

    train_loss_history = []
    test_loss_history = []

    train_dataset = get_dataset(train_interactions)
    test_dataset = get_dataset(test_interactions)
    test_users, test_movies, test_ratings = test_dataset.tensors

    data_loader = DataLoader(train_dataset, batch_size=batch_size)

    model.to(DEVICE)

    for epoch in tqdm(range(0, epochs), desc='Training'):
        train_loss = 0

        for users_batch, movies_batch, ratings_batch in data_loader:
            optimizer.zero_grad()

            prediction = model(users_batch, movies_batch)
            loss = loss_func(prediction, ratings_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        test_prediction = model(test_users, test_movies)
        test_loss = loss_func(test_prediction, test_ratings).item()

        train_loss /= len(data_loader)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        if verbose:
            msg = f'Train loss: {train_loss:.3f}, '
            msg += f'Test loss: {test_loss:.3f}'
            tqdm.write(msg)

    return train_loss_history, test_loss_history
