# models.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

class NaiveBayesModel:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0]}
        gs = GridSearchCV(MultinomialNB(), param_grid, cv=cv, scoring="f1")
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        return gs.best_params_, gs.best_score_


class LogisticRegressionModel:
    def __init__(self, C=1.0, max_iter=1000):
        self.model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            max_iter=max_iter
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {"C": [0.01, 0.1, 1, 10]}
        gs = GridSearchCV(
            LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000),
            param_grid, cv=cv, scoring="f1"
        )
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        return gs.best_params_, gs.best_score_


class DecisionTreeModel:
    def __init__(self, max_depth=None, ccp_alpha=0.0):
        self.model = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=ccp_alpha)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def tune(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                "max_depth": [None, 5, 10, 20],
                "ccp_alpha": [0.0, 0.001, 0.01]
            }
        gs = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=cv, scoring="f1")
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        return gs.best_params_, gs.best_score_


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

    def tune(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_features": ["sqrt", "log2", None],
                "max_depth": [None, 10, 20]
            }
        gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                          param_grid, cv=cv, scoring="f1")
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        return gs.best_params_, gs.best_score_


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

    def tune(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            }
        gs = GridSearchCV(GradientBoostingClassifier(random_state=42),
                          param_grid, cv=cv, scoring="f1")
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        return gs.best_params_, gs.best_score_
