import os
import logging


logger = logging.getLogger(__name__)


class RecSysOptions():
    def __init__(self):
        logger.info('started RecSysOptions init started')
        # self.core_directory = 'C:\\Users\\Acer\\Machine Learning\\кусыны ьа натворил хуйни'
        self.core_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # model
        self.model_name = 'model'
        self.model_extention = 'csv'
        self.model_store = None
        self.model_data_path = None

        # warm-up data
        self.train_data_folder = None
        self.test_data_folder = None
        self.items_data_path = None
        self.users_data_path = None

        # training data
        self.train_data_path = None
        self.test_data_path = None

        # training
        self.n_vectors = 30

        # proceeeding
        self.data_loading_sep = '::'
        self.data_loading_engine = 'python'
        self.encoding = 'windows-1251'

        # suprise
        self.rating_scale = (1, 5)
        self.n_epochs = 20

        # user data
        self.credentials = os.environ.get('USER', os.environ.get('USERNAME'))
        self.current_accuracy = 'Model haven\'t been evaluated yet'
        self.datetime_accuracy_test = 'Model haven\'t been evaluated yet'
        self.author = '@nikitalotts'

        self.init_data()
        logger.info('RecSysOptions instance successfully inited')

    def renew_model_name_and_path(self, model_name: str):
        logger.info('started renew_model_name_and_path method')
        self.model_name = model_name
        self.model_data_path = os.path.join(self.model_store, f'{self.model_name}.{self.model_extention}')
        logger.info('renew_model_name_and_path method successfully executed')
        return self.model_path

    def init_data(self):
        logger.info('started init_data method')
        self.model_store = os.path.join(os.path.join(self.core_directory, 'model/store/'))

        if not os.path.exists(self.model_store):
            os.makedirs(self.model_store)
            logger.info(f'created model/store/ directory')

        self.model_data_path = os.path.join(self.model_store, f'{self.model_name}.{self.model_extention}')

        # warm-up data
        self.train_data_folder = os.path.join(os.path.join(self.core_directory, 'data/train/'))
        self.test_data_folder = os.path.join(os.path.join(self.core_directory, 'data/test/'))
        self.items_data_path = os.path.join(os.path.join(self.train_data_folder, 'movies.dat'))
        self.users_data_path = os.path.join(os.path.join(self.train_data_folder, 'users.dat'))

        # training data
        self.train_data_path = os.path.join(os.path.join(self.train_data_folder, 'ratings_train.dat'))
        self.test_data_path = os.path.join(os.path.join(self.test_data_folder, 'ratings_test.dat'))
        logger.info('init_data method successfully executed')


