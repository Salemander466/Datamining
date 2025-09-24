# main.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

from model import NaiveBayesModel, LogisticRegressionModel, DecisionTreeModel, RandomForestModel, GradientBoostingModel
from eval import evaluate_model
import os

def load_data(base_path):

    folds = {}
    classes = {
        "truthful_from_TripAdvisor": 0,   # label 0 = truthful
        "deceptive_from_MTurk": 1        # label 1 = deceptive
    }

    for cls, label in classes.items():
        cls_path = os.path.join(base_path, cls)
        for fold in sorted(os.listdir(cls_path)):
            fold_path = os.path.join(cls_path, fold)
            if not os.path.isdir(fold_path):
                continue
            texts, labels = folds.get(fold, ([], []))
            for fname in os.listdir(fold_path):
                if fname.endswith(".txt"):
                    with open(os.path.join(fold_path, fname), "r", encoding="utf-8") as f:
                        texts.append(f.read().strip())
                        labels.append(label)
            folds[fold] = (texts, labels)

    return folds

def preprocess(train_texts, test_texts, use_bigrams=False):

    ngram_range = (1, 2) if use_bigrams else (1, 1)
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def run_all_models(X_train, y_train, X_test, y_test, class_names):
    """
    Train and evaluate all 5 models.
    """
    results = {}

    # 1. Multinomial Naive Bayes
    nb = NaiveBayesModel()
    nb.tune(X_train, y_train)  # hyperparameter tuning
    nb.train(X_train, y_train)
    print("\n--- Naive Bayes ---")
    results["NaiveBayes"] = evaluate_model(nb.model, X_test, y_test, class_names)

    # 2. Logistic Regression
    lr = LogisticRegressionModel()
    lr.tune(X_train, y_train)
    lr.train(X_train, y_train)
    print("\n--- Logistic Regression ---")
    results["LogisticRegression"] = evaluate_model(lr.model, X_test, y_test, class_names)

    # 3. Decision Tree
    dt = DecisionTreeModel()
    dt.tune(X_train, y_train)
    dt.train(X_train, y_train)
    print("\n--- Decision Tree ---")
    results["DecisionTree"] = evaluate_model(dt.model, X_test, y_test, class_names)

    # 4. Random Forest
    rf = RandomForestModel()
    rf.tune(X_train, y_train)
    rf.train(X_train, y_train)
    print("\n--- Random Forest ---")
    results["RandomForest"] = evaluate_model(rf.model, X_test, y_test, class_names)

    # 5. Gradient Boosting
    gb = GradientBoostingModel()
    gb.tune(X_train, y_train)
    gb.train(X_train, y_train)
    print("\n--- Gradient Boosting ---")
    results["GradientBoosting"] = evaluate_model(gb.model, X_test, y_test, class_names)

    return results


def main():



    # === Step 1: Load your folds ===
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "/Data/op_spam_v1.4/negative_polarity"
    )

    # === Step 1: Load folds ===
    folds = load_data(BASE_DIR)

    # Train on folds 1–4, test on fold 5
    train_texts, train_labels = [], []
    for f in ["fold1", "fold2", "fold3", "fold4"]:
        X, y = folds[f]
        train_texts.extend(X)
        train_labels.extend(y)

    test_texts, test_labels = folds["fold5"]

    # Example split (replace with your fold logic)
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # === Step 2: Preprocess ===
    X_train, X_test, vectorizer = preprocess(X_train_texts, X_test_texts, use_bigrams=False)

    # === Step 3: Train + Evaluate ===
    class_names = ["truthful", "deceptive"]  # adjust as needed
    results = run_all_models(X_train, y_train, X_test, y_test, class_names)

    # === Step 4: Store Results ===
    results_df = pd.DataFrame(results).T
    results_df.to_csv("results_summary.csv")
    print("\nFinal results saved to results_summary.csv")


if __name__ == "__main__":
    main()
