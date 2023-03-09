import logging
from model import RecSysMF
from options import RecSysOptions
from logger import LOG_FILE_PATH

logger = logging.getLogger(__name__)


class Service():
    def __init__(self):
        logger.info('Service instanse init method started')
        self.options = RecSysOptions()
        self.model = RecSysMF(self.options)
        self.model.train()
        logger.info('Service instanse successfully inited')

    # def hello(self, current_file="service.py"):
    #     self.model.current_file = current_file
    #     logger.info('Service hello method successfully executed')
    #     return self.model.current_file

    def train(self):
        logger.info('Service train method started')
        self.model.train()
        logger.info('Service train method successfully executed')

    def evaluate(self):
        logger.info('Service evaluate method started')
        self.model.evaluate()
        logger.info('Service evaluate method successfully executed')

    def predict(self, item_ratings: list, M: int=5):
        logger.info('Service predict method started')
        output = self.model.predict(item_ratings, M)
        logger.info('Service predict method successfully executed')
        return output

    def reload(self):
        logger.info('Service reload method started')
        self.model.warmup()
        logger.info('Service reload method successfully executed')

    def similar(self, movie_name: str, n: int = 10):
        logger.info('Service similair method started')
        output = self.model.get_similar_items(movie_name, n)
        logger.info('Service similair method successfully executed')
        return output

    def log(self, n_rows: int = 20):
        logger.info('Service log method started')
        with open(LOG_FILE_PATH, "r") as log_f:
            logs_rows = log_f.readlines()[-n_rows:]
        logger.info('Service log method successfully executed')
        return logs_rows



