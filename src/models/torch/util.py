import numpy as np
import torch
from torch.utils.data import TensorDataset

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


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
    ratings /= 5

    return TensorDataset(users, movies, ratings)
