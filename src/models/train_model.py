import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def train_with_mlflow(X_train, y_train):
    mlflow.set_tracking_uri("http://mlflow:5000")

    with mlflow.start_run():
        # Define pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])

        # Hyperparameter tuning
        parameters = {
            'tfidf__max_features': [1000, 2000],
            'clf__C': [0.1, 1, 10]
        }

        gs = GridSearchCV(pipeline, parameters, cv=3, scoring='accuracy')
        gs.fit(X_train, y_train)

        # Log parameters and metrics
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("best_score", gs.best_score_)

        # Log model
        mlflow.sklearn.log_model(gs.best_estimator_, "model")

        # Register best model
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "prod_model"
        )

    return gs.best_estimator_