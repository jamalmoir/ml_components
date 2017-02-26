from abc import ABCMeta, abstractmethod, abstractproperty


class Model(metaclass=ABCMeta):
    @abstractproperty
    def model(self):
        pass

    @abstractmethod
    def train(self):
        pass


class Classifier(Model):
    @abstractmethod
    def predict(self):
        pass


class Clusterer(Model):
    @abstractmethod
    def get_labels(self):
        pass
