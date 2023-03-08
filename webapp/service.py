from model import RecSysMF
from options import RecSysOptions


class Service():
    def __init__(self):
        self.options = RecSysOptions()
        self.model = RecSysMF(self.options)
        self.model.train()

    def hello(self, current_file="service.py"):
        self.model.current_file = current_file
        return self.model.current_file

    def train(self):
        self.model.train()

    def evaluate(self):
        self.model.evaluate()

    def predict(self, item_ratings: list, M: int=5):
        output = self.model.predict(item_ratings, M)
        return output

    def reload(self):
        self.model.warmup()

    def similair(self, movie_name: str, n: int = 10):
        output = self.model.get_similar_items(movie_name, n)
        return output



