import mlflow
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV  # Or RandomizedSearchCV, Optuna, etc.
from src.data.data_loader import load_and_preprocess_data  # Import your data loading function

def train_model(X_train, y_train, model_type="logistic_regression", param_space=None):  # Allow different model types
    with mlflow.start_run():
        if model_type == "logistic_regression":  # Example: Logistic Regression
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression(solver='liblinear', max_iter=1000)) # Increased max_iter
            ])
            if not param_space:
                param_space = {
                    'tfidf__max_features': [1000, 2000],
                    'clf__C': [0.1, 1, 10]
                }
        # ... (Add other model types with their respective pipelines and parameter spaces)

        grid_search = GridSearchCV(pipeline, param_grid=param_space, cv=5, scoring='accuracy', n_jobs=-1) # Added n_jobs for parallelization
        grid_search.fit(X_train, y_train)

        # MLflow logging (parameters, metrics, model)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_accuracy", grid_search.best_score_) # More specific metric name
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

        # ... (Register the model as before)
        return grid_search.best_estimator_



# Example usage in a training script
if __name__ == "__main__": # Ensure this code runs only when the script is executed directly

    X_train, _, y_train, _ = load_and_preprocess_data() # Load and preprocess data
    model = train_model(X_train, y_train)  # Call the training function