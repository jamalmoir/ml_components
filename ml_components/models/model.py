from abc import ABCMeta, abstractmethod, abstractproperty


class Model(metaclass=ABCMeta):
    """The base model class."""
    @abstractproperty
    def model(self):
        pass

    @abstractmethod
    def train(self):
        pass


class Classifier(Model):
    """The base classifier class"""
    @abstractmethod
    def predict(self):
        pass


class Clusterer(Model):
    """The base clusterer class."""
    @abstractmethod
    def get_labels(self):
        pass
