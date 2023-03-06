from model import RecSysMF
from options import RecSysOptions


class Service(object):
    def __init__(self):
        self.options = RecSysOptions()
        self.model = RecSysMF(self.options)
        pass

    def hello(self, current_file="service.py"):
        self.model.current_file = current_file
        return self.model.current_file

    def train(self):
        self.model.train()

    def evaluate(self):
        self.model.evaluate()


