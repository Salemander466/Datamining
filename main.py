# main.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from cv_model import ModelCVRunner
from preprocessing import vectorize_text

from model2 import NaiveBayesModel, LogisticRegressionModel, DecisionTreeModel, RandomForestModel, GradientBoostingModel
from eval import evaluate_model
import os
from utils import load_data, save_cv_results_and_histograms

def preprocess(train_texts, test_texts, use_bigrams=False):

    ngram_range = (1, 2) if use_bigrams else (1, 1)
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


# def run_all_models(X_train, y_train, X_test, y_test, class_names):
#     """
#     Train and evaluate all 5 models.
#     """
#     results = {}

#     # 1. Multinomial Naive Bayes
#     nb = NaiveBayesModel()
#     nb.tune(X_train, y_train)  # hyperparameter tuning
#     nb.train(X_train, y_train)
#     print("\n--- Naive Bayes ---")
#     results["NaiveBayes"] = evaluate_model(nb.model, X_test, y_test, class_names)

#     # 2. Logistic Regression
#     lr = LogisticRegressionModel()
#     lr.tune(X_train, y_train)
#     lr.train(X_train, y_train)
#     print("\n--- Logistic Regression ---")
#     results["LogisticRegression"] = evaluate_model(lr.model, X_test, y_test, class_names)

#     # 3. Decision Tree
#     dt = DecisionTreeModel()
#     dt.tune(X_train, y_train)
#     dt.train(X_train, y_train)
#     print("\n--- Decision Tree ---")
#     results["DecisionTree"] = evaluate_model(dt.model, X_test, y_test, class_names)

#     # 4. Random Forest
#     rf = RandomForestModel()
#     rf.tune(X_train, y_train)
#     rf.train(X_train, y_train)
#     print("\n--- Random Forest ---")
#     results["RandomForest"] = evaluate_model(rf.model, X_test, y_test, class_names)

#     # 5. Gradient Boosting
#     gb = GradientBoostingModel()
#     gb.tune(X_train, y_train)
#     gb.train(X_train, y_train)
#     print("\n--- Gradient Boosting ---")
#     results["GradientBoosting"] = evaluate_model(gb.model, X_test, y_test, class_names)

#     return results


def run_all_models(X_train, y_train, X_test, y_test, class_names):
    results = {}

    # 1. Multinomial Naive Bayes
    nb = NaiveBayesModel()
    best_params, best_score = nb.tune(X_train, y_train)
    nb.train(X_train, y_train)
    print("\n--- Naive Bayes ---")
    eval_res = evaluate_model(nb.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["NaiveBayes"] = eval_res

    # 2. Logistic Regression
    lr = LogisticRegressionModel()
    best_params, best_score = lr.tune(X_train, y_train)
    lr.train(X_train, y_train)
    print("\n--- Logistic Regression ---")
    eval_res = evaluate_model(lr.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["LogisticRegression"] = eval_res

    # 3. Decision Tree
    dt = DecisionTreeModel()
    best_params, best_score = dt.tune(X_train, y_train)
    dt.train(X_train, y_train)
    print("\n--- Decision Tree ---")
    eval_res = evaluate_model(dt.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["DecisionTree"] = eval_res

    # 4. Random Forest
    rf = RandomForestModel()
    best_params, best_score = rf.tune(X_train, y_train)
    rf.train(X_train, y_train)
    print("\n--- Random Forest ---")
    eval_res = evaluate_model(rf.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["RandomForest"] = eval_res

    # 5. Gradient Boosting
    gb = GradientBoostingModel()
    best_params, best_score = gb.tune(X_train, y_train)
    gb.train(X_train, y_train)
    print("\n--- Gradient Boosting ---")
    eval_res = evaluate_model(gb.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["GradientBoosting"] = eval_res

    return results









# def main():



#     # === Step 1: Load your folds ===
#     BASE_DIR = os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         "Data/op_spam_v1.4/negative_polarity"
#     )


#     # === Step 1: Load folds ===
#     folds = load_data(BASE_DIR)


#     # Train on folds 1–4, test on fold 5

#     train_texts, train_labels = load_data(
#         base=BASE_DIR,
#         folds=[1, 2, 3, 4]
#     )

#     # Test = fold 5
#     test_texts, test_labels = load_data(
#         base=BASE_DIR,
#         folds=[5]
#     )





#     # Example split (replace with your fold logic)
#     X_train_texts, X_test_texts, y_train, y_test = train_test_split(
#         test_texts, test_labels, test_size=0.2, random_state=42, stratify=test_labels
#     )

#     for use_bigrams in [True, False]:

#         # === Step 2: Preprocess ===
#         X_train, X_test, vectorizer = vectorize_text(X_train_texts, X_test_texts, use_bigrams=use_bigrams)

#         # === Step 3: Train + Evaluate ===
#         class_names = ["truthful", "deceptive"]  # adjust as needed
#         results = run_all_models(X_train, y_train, X_test, y_test, class_names)



#         # === Step 4: Store final holdout results (fold 5) ===
#         results_df = pd.DataFrame(results).T
#         results_df.to_csv("results_summary.csv")
#         print("\nFinal results saved to results_summary.csv")

#         # === Step 5: Cross-validation with best params ===
#         runner = ModelCVRunner(cv=5, use_oob_rf=True)
#         cv_results = {}

#         for model_name, res in results.items():
#             best_params = res["best_params"]

#             # Cross-validation on folds 1–4
#             mean_cv_score = runner.cv_with_best_params(model_name, best_params, X_train, y_train)

#             # Evaluate locked model on fold 5
#             eval_res = evaluate_model(
#                 runner.models[model_name],
#                 X_test,
#                 y_test,
#                 class_names=class_names,
#                 model_name=f"{model_name}_cv"
#             )

#             # Collect both CV + holdout results
#             cv_results[model_name] = {
#                 "cv_f1_mean": mean_cv_score,
#                 "best_params": best_params,
#                 "test_accuracy": eval_res["accuracy"],
#                 "test_precision_macro": eval_res["precision_macro"],
#                 "test_recall_macro": eval_res["recall_macro"],
#                 "test_f1_macro": eval_res["f1_macro"],
#             }


#         # Save CV results
#         cv_results_df = pd.DataFrame(cv_results).T
#         cv_results_df.to_csv("cv_results_summary.csv")
#         print("\nCross-validation results saved to cv_results_summary.csv")

# if __name__ == "__main__":
#     main()

def main():
    # === Step 1: Load your folds ===
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Data/op_spam_v1.4/negative_polarity"
    )

    # Train = folds 1–4
    train_texts, train_labels = load_data(base=BASE_DIR, folds=[1, 2, 3, 4])

    # Test = fold 5
    test_texts, test_labels = load_data(base=BASE_DIR, folds=[5])

    # Loop over unigrams & bigrams
    for use_bigrams in [False, True]:
        print("\n" + "="*50)
        print(f" Running with {'unigrams + bigrams' if use_bigrams else 'unigrams only'} ")
        print("="*50)

        # === Step 2: Preprocess ===
        X_train, X_test, vectorizer = vectorize_text(
            train_texts, test_texts, use_bigrams=use_bigrams
        )

        # === Step 3: Train + Evaluate ===
        class_names = ["truthful", "deceptive"]
        results = run_all_models(X_train, train_labels, X_test, test_labels, class_names)

        # === Step 4: Save holdout results ===
        suffix = "bigrams" if use_bigrams else "unigrams"
        results_df = pd.DataFrame(results).T
        results_df.to_csv(f"results_summary_{suffix}.csv")
        print(f"\n✅ Final results saved to results_summary_{suffix}.csv")

        # === Step 5: Cross-validation with best params ===
        runner = ModelCVRunner(cv=5, use_oob_rf=True)
        cv_results = {}

        for model_name, res in results.items():
            best_params = res["best_params"]

            # Cross-validation on folds 1–4
            mean_cv_score = runner.cv_with_best_params(
                model_name, best_params, X_train, train_labels
            )

            # Evaluate locked model on fold 5
            eval_res = evaluate_model(
                runner.models[model_name],
                X_test,
                test_labels,
                class_names=class_names,
                model_name=f"{model_name}_{suffix}_cv"
            )

            # Collect both CV + holdout results
            cv_results[model_name] = {
                "cv_f1_mean": mean_cv_score,
                "best_params": best_params,
                "test_accuracy": eval_res["accuracy"],
                "test_precision_macro": eval_res["precision_macro"],
                "test_recall_macro": eval_res["recall_macro"],
                "test_f1_macro": eval_res["f1_macro"],
            }


        # Save CV results
        cv_results_df = pd.DataFrame(cv_results).T
        cv_results_df.to_csv(f"cv_results_summary_{suffix}.csv")
        print(f"✅ Cross-validation results saved to cv_results_summary_{suffix}.csv")
        save_cv_results_and_histograms(cv_results, suffix)

if __name__ == "__main__":
    main()

