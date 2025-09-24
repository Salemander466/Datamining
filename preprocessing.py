from sklearn.feature_extraction.text import CountVectorizer

def vectorize_text(X_train, X_test, use_bigrams=False):
    """
    Make list of matrices using unigrams
    and unigrams + bigrams
    """

    ngram_range = (1, 2) if use_bigrams else (1, 1)

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=ngram_range,
        max_df=0.95,
        min_df=2,
    )


    # Train vectorizer and apply test
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, vectorizer