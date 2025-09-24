# cv_models.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

class BaseCVModel:
    """Base class with consistent interface for CV training."""
    def __init__(self):
        self.model = None
        self.grid_search = None

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def best_params(self):
        return self.grid_search.best_params_ if self.grid_search else None

    def best_score(self):
        return self.grid_search.best_score_ if self.grid_search else None


class NaiveBayesCV(BaseCVModel):
    def cv_train(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0]}
        self.grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=cv, scoring="f1")
        self.grid_search.fit(X_train, y_train)
        self.model = self.grid_search.best_estimator_


class LogisticRegressionCVModel(BaseCVModel):
    def cv_train(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {"C": [0.01, 0.1, 1, 10]}
        self.grid_search = GridSearchCV(
            LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000),
            param_grid, cv=cv, scoring="f1"
        )
        self.grid_search.fit(X_train, y_train)
        self.model = self.grid_search.best_estimator_


class DecisionTreeCV(BaseCVModel):
    def cv_train(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                "max_depth": [None, 5, 10, 20],
                "ccp_alpha": [0.0, 0.001, 0.01]
            }
        self.grid_search = GridSearchCV(
            DecisionTreeClassifier(), param_grid, cv=cv, scoring="f1"
        )
        self.grid_search.fit(X_train, y_train)
        self.model = self.grid_search.best_estimator_


class RandomForestCV(BaseCVModel):
    def cv_train(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_features": ["sqrt", "log2", None],
                "max_depth": [None, 10, 20]
            }
        self.grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid, cv=cv, scoring="f1"
        )
        self.grid_search.fit(X_train, y_train)
        self.model = self.grid_search.best_estimator_


class GradientBoostingCV(BaseCVModel):
    def cv_train(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            }
        self.grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid, cv=cv, scoring="f1"
        )
        self.grid_search.fit(X_train, y_train)
        self.model = self.grid_search.best_estimator_
