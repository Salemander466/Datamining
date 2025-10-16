import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data(base="data\\op_spam_v1.4\\negative_polarity", folds=[1,2,3,4]):
    texts = []
    labels = []

    sources = [
        ("truthful_from_Web", 1),
        ("deceptive_from_MTurk", 0)
    ]

    for folder, label in sources:
        for fold in folds:
            fold_path = os.path.join(base, folder, f"fold{fold}")
            print(f"FOLD PATH: {fold_path}")
            for filename in os.listdir(fold_path):
                file_path = os.path.join(fold_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read().strip())
                    labels.append(label)

    return texts, labels



def save_cv_results_and_histograms(cv_results, suffix=""):

    #Create main output directory
    output_dir = f"cv_analysis_{suffix}"
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(cv_results, dict):
        cv_results_df = pd.DataFrame(cv_results).T
    else:
        cv_results_df = cv_results.copy()

    # Save CSV
    csv_path = os.path.join(output_dir, f"cv_results_summary_{suffix}.csv")
    cv_results_df.to_csv(csv_path)
    print(f"Cross-validation results saved to {csv_path}")
    metrics = [
        "cv_f1_mean",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
    ]

    #Plot style
    sns.set(style="whitegrid", context="talk")

    #Create subfolder for histograms
    hist_folder = os.path.join(output_dir, "histograms")
    os.makedirs(hist_folder, exist_ok=True)

    for metric in metrics:
        if metric not in cv_results_df.columns:
            continue
        data = cv_results_df[metric].dropna()

        plt.figure(figsize=(8, 6))
        sns.histplot(data, bins=10, kde=True, color="steelblue", edgecolor="black")



        plt.title(f"{metric.replace('_', ' ').title()} Across Models", fontsize=16, weight="bold")
        plt.xlabel(metric.replace('_', ' ').title(), fontsize=13)
        plt.ylabel("Number of Models", fontsize=13)

        #Summary statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()

        plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.3f}")
        plt.axvline(median_val, color="green", linestyle=":", linewidth=2, label=f"Median = {median_val:.3f}")

        plt.legend(fontsize=11)
        plt.text(
            0.95, 0.9,
            f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}",
            transform=plt.gca().transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8)
        )

        plt.tight_layout()
        fig_path = os.path.join(hist_folder, f"{metric}_histogram.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

    print(f"Histograms saved to {hist_folder}")
    print("Each histogram shows how models compare in terms of the selected metric.")



if __name__ == "__main__":
    texts, labels = load_data()
    print(f"Loaded {len(texts)} documents.")
