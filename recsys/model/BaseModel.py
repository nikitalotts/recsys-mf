import os

from options import RecSysOptions
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, n_vectors: int = 5):
        self.n_vectors = n_vectors
        self.data = None

    @abstractmethod
    def load_data(self, options: RecSysOptions):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def save(self, name: str, options: RecSysOptions):
        pass