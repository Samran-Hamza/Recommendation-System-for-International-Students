import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class RecommendationEngine:
    def __init__(self, user_features, item_features):
        """
        Initialize the recommendation engine with user and item features.
        """
        self.user_features = user_features
        self.item_features = item_features

    def recommend_items(self, user_id, top_n=5):
        """
        Recommend top-N items for a given user based on cosine similarity.
        """
        user_vector = self.user_features[user_id].reshape(1, -1)
        similarities = cosine_similarity(user_vector, self.item_features).flatten()
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return top_indices, similarities[top_indices]

    def train_matrix_factorization(self, interaction_matrix, latent_features=20):
        """
        Train a matrix factorization model using Non-Negative Matrix Factorization (NMF).
        """
        model = NMF(n_components=latent_features, init='random', random_state=42)
        W = model.fit_transform(interaction_matrix)
        H = model.components_
        return W, H

    def ensemble_recommendations(self, X_train, y_train):
        """
        Train an ensemble of models for recommendations.
        """
        clf = MultinomialNB(alpha=0.1)
        sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
        lgb = LGBMClassifier(
            n_iter=2500,
            verbose=-1,
            objective='cross_entropy',
            metric='auc',
            learning_rate=0.01,
            colsample_bytree=0.78,
            lambda_l1=4.56,
            lambda_l2=2.97,
            min_data_in_leaf=115,
            max_depth=23,
            max_bin=898
        )
        cat = CatBoostClassifier(
            iterations=2000,
            verbose=0,
            l2_leaf_reg=6.65,
            learning_rate=0.1,
            subsample=0.4,
            allow_const_label=True,
            loss_function='CrossEntropy'
        )
        weights = [0.068, 0.311, 0.31, 0.311]
        ensemble = VotingClassifier(
            estimators=[('mnb', clf), ('sgd', sgd_model), ('lgb', lgb), ('cat', cat)],
            weights=weights,
            voting='soft',
            n_jobs=-1
        )
        ensemble.fit(X_train, y_train)
        return ensemble

# Пример использования
if __name__ == "__main__":
    # данные взаимодействий пользователей и вузов
    interaction_matrix = np.random.rand(10, 20)  # 10 пользователей, 20 вузов

    engine = RecommendationEngine(None, None)
    user_features, item_features = engine.train_matrix_factorization(interaction_matrix)

    recommendations, scores = engine.recommend_items(user_id=0, top_n=5)
    print("Recommended Items:", recommendations)
    print("Scores:", scores)