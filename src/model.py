import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self.load()

    def train(self, X: np.ndarray, y: np.ndarray):
        self._model = RandomForestClassifier(max_features=5, min_samples_leaf=10, n_estimators=500,n_jobs=-1, random_state=0)
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self):
        if self._model is not None:
            joblib.dump(self._model, self._model_path)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        try:
            self._model = joblib.load(self._model_path)
        except:
            self._model = None
        return self


# model_path = Path(__file__).parent / "model.joblib"
# model = Model(model_path)


def get_model():
    return model


# if __name__ == "__main__":
#     X, y = load_boston(return_X_y=True)
#     model.train(X, y)
#     model.save()