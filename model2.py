# fast_model.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


class NaiveBayesModel:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, X_train, y_train, param_distributions=None, cv=3, n_iter=10):
        if param_distributions is None:
            #Hyper Parameters
            param_distributions = {
                "alpha": np.linspace(0.1, 2.0, 10),
            }
        rs = RandomizedSearchCV(
            MultinomialNB(),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42
        )
        rs.fit(X_train, y_train)
        self.model = rs.best_estimator_
        return rs.best_params_, rs.best_score_


class LogisticRegressionModel:
    def __init__(self, C=1.0, max_iter=1000):
        self.model = LogisticRegression(
            penalty="l1", solver="liblinear", C=C, max_iter=max_iter
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, X_train, y_train, param_distributions=None, cv=3, n_iter=10):
        if param_distributions is None:
            #Hyper Parameters
            param_distributions = {"C": np.logspace(-2, 2, 20)}
        rs = RandomizedSearchCV(
            LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42
        )
        rs.fit(X_train, y_train)
        self.model = rs.best_estimator_
        return rs.best_params_, rs.best_score_


class DecisionTreeModel:
    def __init__(self, max_depth=None, ccp_alpha=0.0):
        self.model = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=ccp_alpha)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, X_train, y_train, param_distributions=None, cv=3, n_iter=10):
        if param_distributions is None:
            #Hyper Parameters
            param_distributions = {
                "max_depth": [None, 5, 10, 20],
                "ccp_alpha": np.linspace(0.0, 0.02, 5),
            }
        rs = RandomizedSearchCV(
            DecisionTreeClassifier(),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42
        )
        rs.fit(X_train, y_train)
        self.model = rs.best_estimator_
        return rs.best_params_, rs.best_score_


class RandomForestModel:
    def __init__(self, n_estimators=100, max_features="sqrt", random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, X_train, y_train, param_distributions=None, cv=3, n_iter=15):
        if param_distributions is None:
            #Hyper Parameters
            param_distributions = {
                "n_estimators": [100, 200],
                "max_features": ["sqrt", "log2", None],
                "max_depth": [None, 10, 20],
            }
        rs = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42
        )
        rs.fit(X_train, y_train)
        self.model = rs.best_estimator_
        return rs.best_params_, rs.best_score_


class GradientBoostingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, X_train, y_train, param_distributions=None, cv=3, n_iter=15):
        if param_distributions is None:
            #Hyper Parameters
            param_distributions = {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 5],
            }
        rs = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42
        )
        rs.fit(X_train, y_train)
        self.model = rs.best_estimator_
        return rs.best_params_, rs.best_score_
