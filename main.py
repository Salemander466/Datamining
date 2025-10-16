from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from cv_model import ModelCVRunner
from preprocessing import vectorize_text

from model2 import NaiveBayesModel, LogisticRegressionModel, DecisionTreeModel, RandomForestModel, GradientBoostingModel
from eval import evaluate_model
import os
from utils import load_data, save_cv_results_and_histograms



def run_all_models(X_train, y_train, X_test, y_test, class_names):
    results = {}

    #Multinomial Naive Bayes
    nb = NaiveBayesModel()
    best_params, best_score = nb.tune(X_train, y_train)
    nb.train(X_train, y_train)
    print("\n--- Naive Bayes ---")
    eval_res = evaluate_model(nb.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["NaiveBayes"] = eval_res

    #Logistic Regression
    lr = LogisticRegressionModel()
    best_params, best_score = lr.tune(X_train, y_train)
    lr.train(X_train, y_train)
    print("\n--- Logistic Regression ---")
    eval_res = evaluate_model(lr.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["LogisticRegression"] = eval_res

    #Decision Tree
    dt = DecisionTreeModel()
    best_params, best_score = dt.tune(X_train, y_train)
    dt.train(X_train, y_train)
    print("\n--- Decision Tree ---")
    eval_res = evaluate_model(dt.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["DecisionTree"] = eval_res

    #Random Forest
    rf = RandomForestModel()
    best_params, best_score = rf.tune(X_train, y_train)
    rf.train(X_train, y_train)
    print("\n--- Random Forest ---")
    eval_res = evaluate_model(rf.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["RandomForest"] = eval_res

    #Gradient Boosting
    gb = GradientBoostingModel()
    best_params, best_score = gb.tune(X_train, y_train)
    gb.train(X_train, y_train)
    print("\n--- Gradient Boosting ---")
    eval_res = evaluate_model(gb.model, X_test, y_test, class_names)
    eval_res["best_params"] = best_params
    eval_res["cv_score"] = best_score
    results["GradientBoosting"] = eval_res

    return results



def main():
    #Load your folds
    BASE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Data/op_spam_v1.4/negative_polarity"
    )

    #Train with folds 1–4
    train_texts, train_labels = load_data(base=BASE_DIR, folds=[1, 2, 3, 4])

    #Test with fold 5
    test_texts, test_labels = load_data(base=BASE_DIR, folds=[5])

    #Looping through unigrams & bigrams
    for use_bigrams in [False, True]:
        print("\n" + "="*50)
        print(f" Running with {'unigrams + bigrams' if use_bigrams else 'unigrams only'} ")
        print("="*50)

        #Preprocess
        X_train, X_test, vectorizer = vectorize_text(
            train_texts, test_texts, use_bigrams=use_bigrams
        )

        #Train & Evaluate
        class_names = ["truthful", "deceptive"]
        results = run_all_models(X_train, train_labels, X_test, test_labels, class_names)

        #Save holdout results
        suffix = "bigrams" if use_bigrams else "unigrams"
        results_df = pd.DataFrame(results).T
        results_df.to_csv(f"results_summary_{suffix}.csv")
        print(f"\n✅ Final results saved to results_summary_{suffix}.csv")

        #Cross validation with best params
        runner = ModelCVRunner(cv=5, use_oob_rf=True)
        cv_results = {}

        for model_name, res in results.items():
            best_params = res["best_params"]

            #Cross-validation with folds 1–4
            mean_cv_score = runner.cv_with_best_params(
                model_name, best_params, X_train, train_labels
            )

            #Evaluate with fold 5
            eval_res = evaluate_model(
                runner.models[model_name],
                X_test,
                test_labels,
                class_names=class_names,
                model_name=f"{model_name}_{suffix}_cv"
            )

            #Collect both CV
            cv_results[model_name] = {
                "cv_f1_mean": mean_cv_score,
                "best_params": best_params,
                "test_accuracy": eval_res["accuracy"],
                "test_precision_macro": eval_res["precision_macro"],
                "test_recall_macro": eval_res["recall_macro"],
                "test_f1_macro": eval_res["f1_macro"],
            }


        #Save CV results
        cv_results_df = pd.DataFrame(cv_results).T
        cv_results_df.to_csv(f"cv_results_summary_{suffix}.csv")
        print(f"✅ Cross-validation results saved to cv_results_summary_{suffix}.csv")
        save_cv_results_and_histograms(cv_results, suffix)

if __name__ == "__main__":
    main()

