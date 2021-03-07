from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.models.matrix_factorization import MatrixFactorization
from src.models.recommender import RecommenderSystem

RATINGS_PATH = Path('data/ratings_small.csv')
# RATINGS_PATH = Path('data/ratings.csv')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_interactions(
    ratings: pd.DataFrame,
    user_encoder: LabelEncoder,
    movie_encoder: LabelEncoder
) -> np.ndarray:

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


def train(
    model: MatrixFactorization,
    epochs: int = 100,
    batch_size: int = 200
):
    loss_func = nn.MSELoss()

    optimizer = optim.SparseAdam(model.parameters(), lr=1e-3)

    loss_history = []
    users, movies = model.interactions.nonzero()
    ratings = model.interactions[users, movies]

    permutation = torch.randperm(len(users))

    users = users[permutation]
    movies = movies[permutation]
    ratings = ratings[permutation]

    users = torch.LongTensor(users).to(device)
    movies = torch.LongTensor(movies).to(device)
    ratings = torch.FloatTensor(ratings).to(device)

    dataset = TensorDataset(users, movies, ratings)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    model.to(device)

    for epoch in tqdm(range(0, epochs), desc='Training'):
        loss_cum = 0

        for users_batch, movies_batch, ratings_batch in data_loader:
            optimizer.zero_grad()

            prediction = model(users_batch, movies_batch)
            loss = loss_func(prediction, ratings_batch)

            loss.backward()
            optimizer.step()

            loss_cum += loss.item()

        loss_history.append(loss_cum)
        tqdm.write(f'Loss: {loss_cum}')


def test_model(model: RecommenderSystem, test_ratings: pd.DataFrame):
    test_users = set(test_ratings['userId'].values)

    total_pred_scores = []
    total_test_scores = []

    for user_id in tqdm(test_users, desc='Testing predictions'):
        pred_movies, pred_scores = model.predict_scores(user_id)

        pred_movies = {
            pred_movie: pred_score
            for pred_movie, pred_score
            in zip(pred_movies, pred_scores)
        }

        user_ratings = test_ratings.loc[test_ratings['userId'] == user_id]
        test_scores = list(user_ratings['rating'].values)

        pred_scores = [
            # ! TODO: co tu zrobić, jak nie ma filmu? :(
            pred_movies.get(movie_id, 0) for movie_id
            in user_ratings['movieId'].values
        ]

        total_pred_scores += pred_scores
        total_test_scores += test_scores

    r2 = r2_score(total_test_scores, total_pred_scores)
    tqdm.write(f'r2 score: {r2}')


def main():

    ratings = pd.read_csv(RATINGS_PATH)

    user_encoder = LabelEncoder()
    user_encoder.fit(ratings['userId'].values)

    movie_encoder = LabelEncoder()
    movie_encoder.fit(ratings['movieId'].values)

    train_ratings, test_ratings = train_test_split(
        ratings,
        test_size=0.1,
        stratify=ratings['userId'].values,
        random_state=42
    )

    interactions = get_interactions(train_ratings, user_encoder, movie_encoder)

    model = MatrixFactorization(
        interactions,
        n_factors=50,
        user_encoder=user_encoder,
        movie_encoder=movie_encoder,
    )

    train(model, epochs=500, batch_size=200)

    with torch.no_grad():
        test_model(model, test_ratings)


if __name__ == '__main__':
    main()
