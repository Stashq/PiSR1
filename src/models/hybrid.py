from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from lightfm import LightFM
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import LabelEncoder
from src.models.recommender import RecommenderSystem


class HybridRecommenderSystem(RecommenderSystem):

    def __init__(
        self,
        user_encoder: LabelEncoder,
        movie_encoder: LabelEncoder
    ):
        super(RecommenderSystem, self).__init__()

        self.USER_DIM: int
        self.MOVIE_DIM: int

        self.interactions: coo_matrix
        self.movie_features: csr_matrix

        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder

        self.model = LightFM()

    def fit(
        self,
        interactions: coo_matrix,
        movie_features: csr_matrix,
        epochs: int,
        num_threads: int
    ):
        self.USER_DIM, self.MOVIE_DIM = interactions.shape
        self.interactions = interactions
        self.movie_features = movie_features

        self.model.fit(
            interactions,
            item_features=movie_features,
            epochs=5,
            num_threads=2
        )

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
        # self.model.predict(user_id, item_ids, item_features=None, user_features=None, num_threads=1)
        pass

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
        pass

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
        user_id = int(user_id)

        movies = set(range(self.MOVIE_DIM))
        movies_seen = self.interactions.getrow(user_id).toarray().squeeze()
        movies_seen = movies_seen.nonzero()[0]
        movies -= set(movies_seen)
        movies = list(movies)

        movie_features = self.movie_features.toarray()[movies]
        movie_features = csr_matrix(movie_features)

        user_id = np.array([user_id] * len(movies))
        movies = np.array(movies)
        # movies = np.array([0])

        # ratings = self.model.predict(
        #     user_ids=user_id,
        #     item_ids=movies
        # )

        ratings = self.model.predict(
            user_ids=np.array([0, 0, 0]),
            item_ids=np.array([1, 2, 3])
        )


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
