from abc import ABC
from typing import List


class RecommenderSystem(ABC):

    def __init__(self):
        super(RecommenderSystem, self).__init__()

    def predict(user_id: int) -> List[int]:
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

    def predict_score(user_id: int, movie_id: int) -> float:
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
            Predicted movie's score in range [0, 5]
        """
        pass

    def predict_scores(user_id: int) -> float:
        """
        Predicts scores for all the movies, that a user would give.

        Parameters
        ----------
        user_id : int
            User's id from the data set.

        Returns
        -------
        float
            Predicted movie's score in range [0, 5].
        """
        pass
