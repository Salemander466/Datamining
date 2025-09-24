# cv_pipeline.py
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class ModelCVRunner:
    def __init__(self, cv=5, use_oob_rf=True):
        self.cv = cv
        self.use_oob_rf = use_oob_rf
        self.models = {}

    def train_naive_bayes(self, X_train, y_train):
        pipe = Pipeline([
            ("select", SelectKBest(chi2)),
            ("nb", MultinomialNB())
        ])
        param_grid = {
            "select__k": [100, 300, 500, "all"],
            "nb__alpha": [0.1, 0.5, 1.0, 2.0]
        }
        gs = GridSearchCV(pipe, param_grid, cv=self.cv, scoring="f1")
        gs.fit(X_train, y_train)
        self.models["NaiveBayes"] = gs.best_estimator_
        return gs.best_params_, gs.best_score_

    def train_logistic_regression(self, X_train, y_train):
        model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
        param_grid = {"C": [0.01, 0.1, 1, 10]}
        gs = GridSearchCV(model, param_grid, cv=self.cv, scoring="f1")
        gs.fit(X_train, y_train)
        self.models["LogisticRegression"] = gs.best_estimator_
        return gs.best_params_, gs.best_score_

    def train_decision_tree(self, X_train, y_train):
        model = DecisionTreeClassifier()
        param_grid = {
            "max_depth": [None, 5, 10, 20],
            "ccp_alpha": [0.0, 0.001, 0.01]
        }
        gs = GridSearchCV(model, param_grid, cv=self.cv, scoring="f1")
        gs.fit(X_train, y_train)
        self.models["DecisionTree"] = gs.best_estimator_
        return gs.best_params_, gs.best_score_

    def train_random_forest(self, X_train, y_train):
        if self.use_oob_rf:
            # OOB evaluation instead of CV
            model = RandomForestClassifier(
                n_estimators=200, max_features="sqrt",
                oob_score=True, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models["RandomForest"] = model
            return {"oob_score": model.oob_score_}, model.oob_score_
        else:
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_features": ["sqrt", "log2", None],
                "max_depth": [None, 10, 20]
            }
            gs = GridSearchCV(model, param_grid, cv=self.cv, scoring="f1")
            gs.fit(X_train, y_train)
            self.models["RandomForest"] = gs.best_estimator_
            return gs.best_params_, gs.best_score_

    def train_gradient_boosting(self, X_train, y_train):
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5]
        }
        gs = GridSearchCV(model, param_grid, cv=self.cv, scoring="f1")
        gs.fit(X_train, y_train)
        self.models["GradientBoosting"] = gs.best_estimator_
        return gs.best_params_, gs.best_score_

    def evaluate_on_holdout(self, X_test, y_test, metrics):
        """Evaluate all locked models on fold 5 using provided metric fns dict."""
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            results[name] = {m_name: m_func(y_test, y_pred) for m_name, m_func in metrics.items()}
        return results
