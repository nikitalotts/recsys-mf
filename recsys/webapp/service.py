"""
Service class
"""

import logging
from model import RecSysMF
from options import RecSysOptions
from logger import LOG_FILE_PATH

logger = logging.getLogger(__name__)


class Service():
    def __init__(self):
        """Service class initialization method"""

        logger.info('Service instanse init method started')
        self.options = RecSysOptions()
        self.model = RecSysMF(self.options)
        self.model.train()
        logger.info('Service instanse successfully inited')

    def train(self):
        """Method that calls model's train method with default model"""

        logger.info('Service train method started')
        self.model.train()
        logger.info('Service train method successfully executed')

    def evaluate(self):
        """Method that calls model's evaluate method with default model"""

        logger.info('Service evaluate method started')
        self.model.evaluate()
        logger.info('Service evaluate method successfully executed')

    def predict(self, item_ratings: list, M: int=5):
        """Method that calls model's predict method

        inputs:
        item_ratings : list # double list of movie names and movie ratings
        M : int # amount of similar movies

        output: double list of movie names and its estimated ratings
        """

        logger.info('Service predict method started')
        output = self.model.predict(item_ratings, M)
        logger.info('Service predict method successfully executed')
        return output

    def reload(self):
        """Method that calls model's warmup method to reset model state"""
        logger.info('Service reload method started')
        self.model.warmup()
        logger.info('Service reload method successfully executed')

    def similar(self, movie_name: str, n: int = 10):
        """Method that calls model's get_similar_items method

        inputs:
        movie_name : string # name of the movie to find similar to it
        n : int # amount of similar movies

        output: double list of movie names and its estimated ratings
        """

        logger.info('Service similair method started')
        output = self.model.get_similar_items(movie_name, n)
        logger.info('Service similair method successfully executed')
        return output

    def log(self, n_rows: int = 20):
        """Method that calls return lost n_rows(by default, 20) rows of log-file"""
        logger.info('Service log method started')
        with open(LOG_FILE_PATH, "r", encoding='windows-1251') as log_f:
            logs_rows = log_f.readlines()[-n_rows:]
        logger.info('Service log method successfully executed')
        return logs_rows

    def surprise_evaluate(self):
        """Method that calls model's surprise_train and surprise_evaluate methods
        to renew models state

        inputs:
        None

        output: None
        """
        logger.info('Service surprise_evaluate method started')
        self.model.surprise_train()
        self.model.surprise_evaluate()
        logger.info('Service surprise_evaluate method successfully executed')

    def info(self):
        """Method that calls model's sget_info method
        to get and return info about actual model's state

        inputs:
        None

        output: dict
        """
        logger.info('Service info method started')
        info = self.model.get_info()
        logger.info('Service info method successfully executed')
        return info



