import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import os



def evaluate_model(model, X_test, y_test, class_names=None, plot_cm=True, model_name="Model"):
    # Predictions
    y_pred = model.predict(X_test)

    #Performance Metrics
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    }

    print(f"\n=== Evaluation Results ({model_name}) ===")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision (macro): {results['precision_macro']:.4f}")
    print(f"Recall (macro):    {results['recall_macro']:.4f}")
    print(f"F1-score (macro):  {results['f1_macro']:.4f}\n")

    print("Per-class metrics:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    if plot_cm:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {model_name}")
        os.makedirs("CM", exist_ok=True)
        save_path = os.path.join("CM", f"{model_name}_cm.png")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Confusion matrix saved to {save_path}")

        plt.close()

    return results