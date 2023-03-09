import os


class RecSysOptions():
    def __init__(self):
        self.core_directory = 'C:\\Users\\Acer\\Machine Learning\\кусыны ьа натворил хуйни'

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

        self.init_data()

        # print('core_directory', self.core_directory)
        # print('current', os.getcwd())


    def renew_model_name_and_path(self, model_name: str):
        self.model_name = model_name
        self.model_data_path = os.path.join(self.model_store, f'{self.model_name}.{self.model_extention}')
        return self.model_path

    def init_data(self):
        self.model_store = os.path.join(os.path.join(self.core_directory, 'model/store/'))
        self.model_data_path = os.path.join(self.model_store, f'{self.model_name}.{self.model_extention}')

        # warm-up data
        self.train_data_folder = os.path.join(os.path.join(self.core_directory, 'data/train/'))
        self.test_data_folder = os.path.join(os.path.join(self.core_directory, 'data/test/'))
        self.items_data_path = os.path.join(os.path.join(self.train_data_folder, 'movies.dat'))
        self.users_data_path = os.path.join(os.path.join(self.train_data_folder, 'users.dat'))

        # training data
        self.train_data_path = os.path.join(os.path.join(self.train_data_folder, 'ratings_train.dat'))
        self.test_data_path = os.path.join(os.path.join(self.test_data_folder, 'ratings_test.dat'))



    # ages_map = {1: 'Under 18',
    #         18: '18 - 24',
    #         25: '25 - 34',
    #         35: '35 - 44',
    #         45: '45 - 49',
    #         50: '50 - 55',
    #         56: '56+'}

    # occupations_map = {0: 'Not specified',
    #                    1: 'Academic / Educator',
    #                    2: 'Artist',
    #                    3: 'Clerical / Admin',
    #                    4: 'College / Grad Student',
    #                    5: 'Customer Service',
    #                    6: 'Doctor / Health Care',
    #                    7: 'Executive / Managerial',
    #                    8: 'Farmer',
    #                    9: 'Homemaker',
    #                    10: 'K-12 student',
    #                    11: 'Lawyer',
    #                    12: 'Programmer',
    #                    13: 'Retired',
    #                    14: 'Sales / Marketing',
    #                    15: 'Scientist',
    #                    16: 'Self-Employed',
    #                    17: 'Technician / Engineer',
    #                    18: 'Tradesman / Craftsman',
    #                    19: 'Unemployed',
    #                    20: 'Writer'}

    # gender_map = {'M': 'Male', 'F': 'Female'}
        



