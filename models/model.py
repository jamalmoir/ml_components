import abc

class Model:
    __metaclass__ = abc.ABCMeta

    def __ini__(selfi, X, y):
        self.X = X
        self.y = y

    @abc.abstractmethod
    def train():
        return

    @abc.abstractmethod
    def predict():
        return

    @abc.abstractmethod
    def getCost():
        return
