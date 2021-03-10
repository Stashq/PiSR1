from typing import List

import numpy as np
from src.models.torch.util import get_dataset
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


class Trainer:

    def __init__(
        self,
        loss: _Loss,
        regularizers: List[_Loss],
        lr: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        verbose: int = 0
    ):
        super(Trainer, self).__init__()

        self.loss = loss
        self.regularizers = regularizers

        self.LR = lr
        self.WEIGHT_DECAY = weight_decay
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.VERBOSE = verbose

    def train(
        self,
        model: nn.Module,
        train_interactions: np.ndarray,
        test_interactions: np.ndarray,
        is_sparse: bool
    ):

        optimizer: optim.Optimizer

        if is_sparse:
            optimizer = optim.SparseAdam(model.parameters(), lr=self.LR)
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.LR,
                weight_decay=self.WEIGHT_DECAY
            )

        train_loss_history = []
        test_loss_history = []

        train_dataset = get_dataset(train_interactions)
        test_dataset = get_dataset(test_interactions)
        test_users, test_movies, test_ratings = test_dataset.tensors

        data_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE)

        model.to(DEVICE)

        for epoch in tqdm(range(0, self.EPOCHS), desc='Training'):
            train_loss = 0

            for users_batch, movies_batch, ratings_batch in data_loader:
                optimizer.zero_grad()

                prediction = model(users_batch, movies_batch)
                loss = self.loss(prediction, ratings_batch)

                for regularizer in self.regularizers:
                    loss += regularizer(prediction)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            test_prediction = model(test_users, test_movies)
            test_loss = self.loss(test_prediction, test_ratings).item()
            for regularizer in self.regularizers:
                test_loss += regularizer(test_prediction).item()

            train_loss /= len(data_loader)

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)

            if self.VERBOSE:
                msg = f'Train loss: {train_loss:.3f}, '
                msg += f'Test loss: {test_loss:.3f}'
                tqdm.write(msg)

        return train_loss_history, test_loss_history
