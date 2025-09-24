# cv_pipeline.py
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class ModelCVRunner:
    def __init__(self, cv=5, use_oob_rf=True):
        self.cv = cv
        self.use_oob_rf = use_oob_rf
        self.models = {}

    def cv_with_best_params(self, model_name, best_params, X_train, y_train):

        if model_name == "NaiveBayes":
            # build pipeline with feature selection
            k = best_params.get("select__k", "all")
            alpha = best_params.get("nb__alpha", 1.0)
            model = Pipeline([
                ("select", SelectKBest(chi2, k=k)),
                ("nb", MultinomialNB(alpha=alpha))
            ])

        elif model_name == "LogisticRegression":
            C = best_params.get("C", 1.0)
            model = LogisticRegression(
                penalty="l1", solver="liblinear", max_iter=1000, C=C
            )

        elif model_name == "DecisionTree":
            model = DecisionTreeClassifier(
                max_depth=best_params.get("max_depth", None),
                ccp_alpha=best_params.get("ccp_alpha", 0.0)
            )

        elif model_name == "RandomForest":
            if self.use_oob_rf:
                # OOB evaluation
                model = RandomForestClassifier(
                    n_estimators=best_params.get("n_estimators", 200),
                    max_features=best_params.get("max_features", "sqrt"),
                    max_depth=best_params.get("max_depth", None),
                    oob_score=True, random_state=42, n_jobs=-1
                )
                model.fit(X_train, y_train)
                self.models["RandomForest"] = model
                return model.oob_score_
            else:
                model = RandomForestClassifier(
                    n_estimators=best_params.get("n_estimators", 200),
                    max_features=best_params.get("max_features", "sqrt"),
                    max_depth=best_params.get("max_depth", None),
                    random_state=42, n_jobs=-1
                )

        elif model_name == "GradientBoosting":
            model = GradientBoostingClassifier(
                n_estimators=best_params.get("n_estimators", 100),
                learning_rate=best_params.get("learning_rate", 0.1),
                max_depth=best_params.get("max_depth", 3),  # if your sklearn supports it
                random_state=42
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # run cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring="f1_macro")

        # Fit on all training folds so model is ready for fold 5
        model.fit(X_train, y_train)

        self.models[model_name] = model
        return np.mean(scores)

