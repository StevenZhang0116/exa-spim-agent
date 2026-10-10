"""Editable example, not a model whitelist. Replace either function as needed.

This tabular example fits GT-labeled TRAIN rows (labels 1 / 0) and predicts from frozen model files.
For optional raw 3D inputs, see image_context_guide.md and add the images keyword
argument to both functions together with a LOCAL_IMAGE declaration.
Use params for values to tune without changing the training program.
"""

def fit(X_train, y_train, artifact_dir, params):
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          StandardScaler(),
                          LogisticRegression(C=params.get('C', 1.0), max_iter=1000,
                                             class_weight='balanced', random_state=0))
    model.fit(X_train, y_train)
    joblib.dump(model, artifact_dir / 'model.joblib')


def predict(X, artifact_dir, params):
    import joblib
    model = joblib.load(artifact_dir / 'model.joblib')
    return model.predict_proba(X)[:, 1]
