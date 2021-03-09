from abc import ABC
from typing import List, Tuple


class RecommenderSystem(ABC):

    def __init__(self):
        super(RecommenderSystem, self).__init__()
        self.MAX_RATING = 5
        self.MIN_RATING = 0

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
        pass
