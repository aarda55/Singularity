import numpy as np
import pickle
import Model

class MultiModel:
    def __init__(self, models):
        self.models = models

    def predict_multi(self, X):
        outputs = []

        for model in self.models:
            output = model.predict(X)
            outputs.append(output)

        return outputs
    
    def save_multi(self, paths):
        for i, model in enumerate(self.models):
            model.savem(paths[i])

    @classmethod
    def load_multi(cls, paths):
        models = []

        for path in paths:
            model = Model.loadm(path) 
            models.append(model)

        return cls(models)