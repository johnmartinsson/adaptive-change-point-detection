import torch

class BaseActiveLearner():
    """
    The base class for an active learning model.
    """
    def __init__(self):
        pass

    def predict(self, x):
        """
        Hard classification of the input.
        """
        return torch.argmax(self.predict_proba(x))

    def predict_proba(self, x):
        """
        Predict pseudo-probabilities of the input belonging to each class.
        """
        pass

    def predict_query_timings(self, active_dataset):
        """
        Return the query timings.
        """
        print("Predict query timings ...")

    def update(self, active_dataset):
        """
        Update the model given the last query batch to the Oracle.
        """
        print("Update active learner ...")
        pass

    def evaluate(self, dataset):
        """
        Evaluate the model on the dataset.
        """
        print("Evaluate active learner ...")
        pass
