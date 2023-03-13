"""
Abstract class for all models
that consist of method that need to be implemented to model work well
"""

import os

from options import RecSysOptions
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, n_vectors: int = 5):
        # For svd-based model - dim of latent vectors
        # for non-svd-based model can be None
        self.n_vectors = n_vectors

        # Predictions data
        self.data = None

    @abstractmethod
    def load_data(self, options: RecSysOptions):
        """Method that load predictions' data from file to model"""
        pass

    @abstractmethod
    def fit(self):
        """Method that train model"""
        pass

    @abstractmethod
    def save(self, name: str, options: RecSysOptions):
        """Method that save predictions' data to file on disk"""
        pass