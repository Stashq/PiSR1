from typing import Dict, List

import pandas as pd
from sklearn.metrics import classification_report
from src.models.recommender import RecommenderSystem
from tqdm.auto import tqdm


def get_jaccard_ranking(
    model: RecommenderSystem,
    test_ratings: pd.DataFrame
) -> List[float]:

    test_users = set(test_ratings['userId'].values)

    jaccard_scores = []

    for user_id in tqdm(test_users, desc='Testing predictions'):
        pred_movies, pred_scores = model.predict_scores(user_id)

        pred_liked_movies = set(pred_movies[pred_scores >= 3.5])

        test_user_ratings = test_ratings.loc[test_ratings['userId'] == user_id]

        test_user_ratings = test_user_ratings.loc[
            test_user_ratings['rating'] >= 3.5
        ]

        test_liked_movies = set(test_user_ratings['movieId'].values)

        if test_liked_movies:
            jaccard_index = len(test_liked_movies & pred_liked_movies)
            jaccard_index /= len(test_liked_movies)

            jaccard_scores.append(jaccard_index)

    return jaccard_scores


def get_ranking_metrics(
    model: RecommenderSystem,
    test_ratings: pd.DataFrame
) -> Dict[str, float]:

    test_users = set(test_ratings['userId'].values)

    test = []
    pred = []

    for user_id in tqdm(test_users, desc='Testing predictions'):
        pred_movies, pred_scores = model.predict_scores(user_id)

        pred_liked_movies = set(pred_movies[pred_scores >= 3.5])

        test_user_ratings = test_ratings.loc[test_ratings['userId'] == user_id]

        test_user_ratings = test_user_ratings.loc[
            test_user_ratings['rating'] >= 3.5
        ]

        test_liked_movies = set(test_user_ratings['movieId'].values)

        for movie_id in test_liked_movies:
            test.append(True)
            pred.append(movie_id in pred_liked_movies)

    return classification_report(test, pred)

