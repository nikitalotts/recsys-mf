"""
RecSysOptions class where stored based settings that used in project
"""

import os
import logging

# get logger instance
logger = logging.getLogger(__name__)


class RecSysOptions():
    def __init__(self):
        logger.info('started RecSysOptions init started')

        # base workdir of the project: string
        self.core_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # name of the file where predictions are stored: string
        self.model_name = 'model'

        # prediction's file extension: string
        self.model_extention = 'csv'

        # path to ./recsys/model/store: string
        self.model_store = None

        # path to ./recsys/model/store/{model_name}.{model_extention}: string
        self.model_data_path = None

        # path to /data/train: string
        self.train_data_folder = None

        # path to /data/test: string
        self.test_data_folder = None

        # path to /data/train/movies.dat: string
        self.items_data_path = None

        # path to /data/train/users.dat: string
        self.users_data_path = None

        # path to /data/train/ratings_train.dat: string
        self.train_data_path = None

        # path to /data/test/ratings_test.dat: string
        self.test_data_path = None

        # dim of latent vectors for svd-based models: int
        self.n_vectors = 30

        # separator symbols that used in ratings data: string
        self.data_loading_sep = '::'

        # engine that used in file reading: string
        self.data_loading_engine = 'python'

        # data encoding: string
        self.encoding = 'windows-1251'

        # sklearn-surprise Scaler scale settings: tuple(min, max)
        self.rating_scale = (1, 5)

        # sklearn-surprise SVD's SGD number of epochs: int
        self.n_epochs = 20

        # path to /credentials.txt which created while dockerfile build: string
        self.credentials_file = os.path.join(os.path.join(self.core_directory, 'credentials.txt'))

        # current model accuracy: float
        self.current_accuracy = 'Model haven\'t been evaluated yet'

        # time of last model's evaluation: datetime
        self.datetime_accuracy_test = 'Model haven\'t been evaluated yet'

        # Github's nickname of the author
        self.author = '@nikitalotts'

        self.init_data()
        logger.info('RecSysOptions instance successfully inited')

    def renew_model_name_and_path(self, model_name: str):
        """
        Method that renew model's name

        inputs:
        model_name: str # new name of the model

        outputs:
        model_data_path: string # new full path to model data file
        """

        logger.info('started renew_model_name_and_path method')
        self.model_name = model_name
        self.model_data_path = os.path.join(self.model_store, f'{self.model_name}.{self.model_extention}')
        logger.info('renew_model_name_and_path method successfully executed')
        return self.model_data_path

    def init_data(self):
        """
        Method that setup options' attributes settings and create folders if it doesn't exist
        """

        logger.info('started init_data method')
        self.model_store = os.path.join(os.path.join(self.core_directory, 'model/store/'))

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)
            logger.info(f'created model/store/ directory')

        self.model_data_path = os.path.join(self.model_store, f'{self.model_name}.{self.model_extention}')
        self.train_data_folder = os.path.join(os.path.join(self.core_directory, 'data/train/'))
        self.test_data_folder = os.path.join(os.path.join(self.core_directory, 'data/test/'))
        self.items_data_path = os.path.join(os.path.join(self.train_data_folder, 'movies.dat'))
        self.users_data_path = os.path.join(os.path.join(self.train_data_folder, 'users.dat'))
        self.train_data_path = os.path.join(os.path.join(self.train_data_folder, 'ratings_train.dat'))
        self.test_data_path = os.path.join(os.path.join(self.test_data_folder, 'ratings_test.dat'))

        logger.info('init_data method successfully executed')


