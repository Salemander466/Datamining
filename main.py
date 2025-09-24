from model import NaiveBayesModel, LogisticRegressionModel, DecisionTreeModel, RandomForestModel, GradientBoostingModel

nb = NaiveBayesModel()
nb.train(X_train, y_train)
y_pred = nb.predict(X_test)
