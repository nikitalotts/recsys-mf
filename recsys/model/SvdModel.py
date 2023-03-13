import os
import logging

import numpy as np
import pandas as pd
import surprise
from scipy.sparse.linalg import svds

from BaseModel import BaseModel
from options import RecSysOptions

logger = logging.getLogger(__name__)


class SvdModel(BaseModel):
    def __init__(self, n_vectors: int = 5):
        self.n_vectors = n_vectors
        self.data = None
        logging.info('SvdModel instance successfully inited')

    def load_data(self, options: RecSysOptions):
        logging.info(f'started load_data method, data_path:{options.model_data_path}')
        model_path_split = os.path.splitext(options.model_data_path)
        options.model_name = model_path_split[0]
        options.model_extention = model_path_split[1][1:]

        if options.model_extention == 'csv':
            self.data = pd.read_csv(options.model_data_path,encoding=options.encoding)
            self.data.columns = [int(x) for x in self.data.columns]
        else:
            logger.error("Wrong model extension")
        logging.info('load_data method successfully executed')
        return

    def fit(self, matrix: pd.DataFrame, n_vectors: int, mean_user_rating: np.ndarray, std_user_rating: np.ndarray): #, std_user_rating: np.ndarray):
        logging.info(f'started fit method')
        # print(matrix)
        # print('n vec', n_vectors)
        # print('n mean', mean_user_rating)
        # print(type(matrix))
        # print(type(n_vectors))
        # print(type(mean_user_rating))
        # print(type(std_user_rating))
        u, sigma, vt = svds(matrix.values, k=n_vectors)
        sigma_diag_matrix = np.diag(sigma)
        # predicted_ratings = np.dot(np.dot(u, sigma_diag_matrix), vt) * std_user_rating + mean_user_rating
        # predicted_ratings = np.dot(np.dot(u, sigma_diag_matrix), vt) + mean_user_rating
        # predicted_ratings = (( u @ sigma_diag_matrix @ vt ) + 1) * mean_user_rating
        predicted_ratings = np.dot(np.dot(u, sigma_diag_matrix), vt) * std_user_rating + mean_user_rating
        self.data = pd.DataFrame(predicted_ratings, columns=matrix.columns)
        logging.info('fit method successfully inited')
        return self.data

    def save(self, name: str, options: RecSysOptions):
        logging.info('started save method')
        if name != options.model_name:
            options.renew_model_name_and_path(name)
        self.data.to_csv(options.model_data_path, index=False)
        logging.info('save method successfully executed')
