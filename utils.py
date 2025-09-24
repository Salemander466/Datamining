import os

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


if __name__ == "__main__":
    texts, labels = load_data()
    print(f"Loaded {len(texts)} documents.")
