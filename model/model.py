from __future__ import absolute_import
import fire
from skimage.metrics import mean_squared_error

from SvdModel import *
from options import RecSysOptions

import fire
import numpy as np
import pandas as pd
import os

from options import RecSysOptions
from datetime import datetime
from sklearn.metrics import mean_squared_error
import surprise
from surprise import Reader, Dataset, SVD

from Levenshtein import distance as levenshtein
from sklearn.metrics.pairwise import cosine_similarity
import regex as re

import warnings
warnings.filterwarnings("ignore")


class RecSysMF(object):
    def __init__(self, options: RecSysOptions):
        self.options = options
        self.model = None
        self.users_matrix = None
        self.items_matrix = None
        self.ratings_train = None
        self.ratings_test = None
        self.user_item_matrix = None
        self.mean_user_rating = None
        self.std_user_rating = None
        self.n_users = None
        self.n_items = None
        self.trained = False
        self.surprise_matrix = None
        self.items_similarity_matrix = None
        # self.creation_date = str(datetime.now())
        pass

    # def warmup(self, model_name: str='model', model_extension: str='csv'):
    def warmup(self, model_type: str = 'SVD'):
        print('start warmup', self.options.model_data_path)
        if model_type == 'SVD':
            self.model = SvdModel()
            print('created  class')
        else:
            raise Exception('Invalid model type!')

        if self.__is_model_exists(self.options.model_data_path):
            print('loaded model')
            self.model.load_data(self.options)
        else:
            print('model dont exist')

        self.users_matrix, self.n_users = self.__load_users_data(self.options.users_data_path)
        self.items_matrix, self.n_items = self.__load_items_data(self.options.items_data_path)

        self.users_matrix = self.__proceed_users(self.users_matrix)
        self.items_matrix = self.__proceed_items(self.items_matrix)
        # logger.info(f"Model: {self.options.model} successfully loaded: {datetime.now()}")

    def train(self, train_data_path: str = None):
        print('start train', self.options.train_data_path)
        self.warmup()

        if train_data_path == None:
            train_data_path = self.options.train_data_path

        self.ratings_train = self.__load_ratings(train_data_path)
        self.user_item_matrix = self.__create_user_item_matrix(self.users_matrix, self.items_matrix, self.ratings_train)

        self.user_item_matrix, self.mean_user_rating, self.std_user_rating = self.__normalize_matrix(
            self.user_item_matrix)
        self.model.fit(self.user_item_matrix, self.options.n_vectors, self.mean_user_rating,
                       self.std_user_rating)  # , self.std_user_rating)
        self.model.save(self.options.model_name, self.options)
        self.trained = True
        print('fitted')
        return

    def __is_model_exists(self, model_data_path: str):
        """
        все модели сохраняются в моделс сторе
        """
        return os.path.exists(model_data_path)

    def __create_user_item_matrix(self, users: pd.DataFrame, items: pd.DataFrame, ratings: pd.DataFrame):
        user_item_rating_dataframe = self.__create_user_item_rating_dataframe(users, items, ratings)
        matrix = user_item_rating_dataframe.pivot(index='user_id', columns='movie_id', values='rating')
        return matrix

    def __create_user_item_rating_dataframe(self, users: pd.DataFrame, items: pd.DataFrame, ratings: pd.DataFrame):
        dataframe = pd.merge(ratings, items, on='movie_id', how='left').merge(users, on='user_id', how='left')
        return dataframe

    def __load_model_data(self, model_path: str):
        model_path_split = os.path.splitext(model_path)
        self.options.model_name = model_path_split[0]
        self.options.model_extention = model_path_split[1][1:]

        if self.options.model_extention == 'csv':
            self.model = pd.read_csv(self.options.model_store + f'{model_path}', encoding=self.options.encoding)
        else:
            raise Exception("Wrong model extension")
        return

    def __load_users_data(self, users_data: str):
        users = pd.read_csv(users_data, names=['user_id', 'gender', 'age', 'occupation', 'zip-code'],
                            sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                            encoding=self.options.encoding)
        n_users = users['user_id'].nunique()

        return users, n_users

    # вынести
    def __load_items_data(self, items_data: str):
        items = pd.read_csv(items_data, names=['movie_id', 'title', 'genres'],
                            sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                            encoding=self.options.encoding)
        n_items = items['movie_id'].nunique()

        return items, n_items

    # def __load_ratings(self, ratings_data_path: str, is_train: bool=True):
    def __load_ratings(self, ratings_data_path: str):
        ratings = pd.read_csv(ratings_data_path, names=['user_id', 'movie_id', 'rating', 'date'],
                              sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                              encoding=self.options.encoding)
        # if is_train:
        #     self.ratings_train = ratings
        # else:
        #     self.ratings_test = ratings
        return ratings

    def __proceed_items(self, items_matrix: pd.DataFrame):
        items_matrix['release_year'] = items_matrix['title'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)
        return items_matrix

    def __proceed_users(self, users_matrix: pd.DataFrame):
        return users_matrix

    def __normalize_matrix(self, matrix: pd.DataFrame):
        # mean_user_rating = np.mean(matrix.values, axis=1).reshape(-1, 1)
        mean_user_rating = np.nanmean(matrix.values, axis=1).reshape(-1, 1)
        std_user_rating = np.nanstd(matrix.values, axis=1).reshape(-1, 1)
        # std_user_rating = np.std(matrix.values, axis=1).reshape(-1, 1)
        # matrix_normalized_values = (matrix.values - mean_user_rating) / mean_user_rating
        matrix_normalized_values = (matrix.values - mean_user_rating) / std_user_rating
        matrix = pd.DataFrame(data=matrix_normalized_values, index=matrix.index, columns=matrix.columns).fillna(0)

        return matrix, mean_user_rating, std_user_rating

    def __normalize_row(self, row: pd.DataFrame):
        mean_user_rating = np.nanmean(row.values, axis=1).reshape(-1, 1)
        std_user_rating = np.nanstd(row.values, axis=1).reshape(-1, 1)
        row_normalized_values = (row.values - mean_user_rating) / std_user_rating
        row = pd.DataFrame(data=row_normalized_values, index=row.index, columns=row.columns).fillna(0)
        return row, mean_user_rating, std_user_rating

    def __get_movies_ids(self, predictions: pd.DataFrame):
        ids = predictions.columns.values
        return [int(x) for x in ids]

    def evaluate(self, test_data_path: str = None):
        if self.trained == False:
            raise Exception('Model not trained!')

        self.warmup()
        if test_data_path == None:
            test_data_path = self.options.test_data_path

        print('test_Data_path', self.options.test_data_path)

        self.ratings_test = self.__load_ratings(test_data_path)

        print('self.ratings_test', self.ratings_test)

        test_dataset = self.__create_user_item_rating_dataframe(self.users_matrix, self.items_matrix, self.ratings_test)

        print('self.calculate_rmse(test_dataset, self.model.data)')
        print(self.model.data, test_dataset)

        rmse = self.calculate_rmse(test_dataset, self.model.data)
        # rmse2 = self.calc_rmse2(self.model.data)

        # print(f'RMSE: {rmse}')
        # print(f'RMSE: {rmse2}')
        return rmse

    def calculate_rmse(self, dataset: pd.DataFrame, preds: pd.DataFrame):
        real_marks = []
        predictions = []
        for index, row in dataset.iterrows():
            user_id = row['user_id'] - 1
            movie_id = row['movie_id']
            rating = row['rating']
            if movie_id in preds.columns:
                real_marks.append(rating)
                predictions.append(preds[movie_id][user_id])

        return mean_squared_error(real_marks, predictions, squared=False)

    def calc_rmse2(self, preds):
        test_dataset = self.__create_user_item_matrix(self.users_matrix, self.items_matrix, self.ratings_test)
        print(test_dataset)
        rmse = np.nanmean((test_dataset.values - preds.values) ** 2)
        print('rmse2', rmse)
        return rmse

    #### surprise ####
    def surprise_train(self, train_data_path: str = None):
        if train_data_path == None:
            train_data_path = self.options.train_data_path

        self.ratings_train = self.__load_ratings(train_data_path)

        dataset = self.__surprise_get_dataset(self.ratings_train)
        self.__surprise_fit_model(dataset)
        self.trained = True
        print('trained')
        return

    def __surprise_get_dataset(self, ratings: pd.DataFrame):
        return Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader=Reader(rating_scale=(1, 5)))

    def __surprise_fit_model(self, dataset: surprise.Dataset):
        self.model = SVD(n_factors=50)
        self.model.fit(dataset.build_full_trainset())

    def __surprise_make_predictions(self, dataset: surprise.Dataset):
        real_marks = []
        predictions = []
        for row in dataset.build_full_trainset().build_testset():
            real_marks.append(row[2])
            predictions.append(self.model.predict(row[0], row[1]).est)

        return np.array(real_marks), np.array(predictions)

    def __surprise_calculate_rmse(self, real: np.matrix, pred: np.matrix):
        return mean_squared_error(real, pred, squared=False)

    def surprise_evaluate(self, test_data_path: str = None):
        if self.trained == False:
            raise Exception('Model not trained!')

        if test_data_path == None:
            test_data_path = self.options.test_data_path

        self.ratings_test = self.__load_ratings(test_data_path)

        dataset = self.__surprise_get_dataset(self.ratings_test)

        real_marks, predictions = self.__surprise_make_predictions(dataset)

        rmse = self.__surprise_calculate_rmse(real_marks, predictions)

        return rmse

    # query handlers

    def __find_item_by_name(self, received_name: str):
        item_index = self.items_matrix['title'].apply(
            lambda title: levenshtein(re.sub(r' \([0-9]{4}\)', '', title.lower()), received_name.lower())).idxmin()
        item_id = self.items_matrix.loc[item_index]['movie_id']
        return item_id

    def __calculate_items_similarity_matrix(self, items_matrix: pd.DataFrame):
        # t тк вектор - фильм
        similarity_matrix = cosine_similarity(items_matrix, items_matrix)
        # нуля чтоб сам себя не рекоммендовал
        np.fill_diagonal(similarity_matrix, 0)
        similarity_df = pd.DataFrame(similarity_matrix, self.model.data.columns, self.model.data.columns)
        return similarity_df

    def __find_similar(self, movie_id: int, n: int = 5):
        items_idxs = np.array(self.items_similarity_matrix[movie_id].sort_values(ascending=False)[:n].index.values,
                              dtype=int).tolist()

        items = self.__sort_items_by_ids(self.items_matrix, items_idxs)

        items_idxs = [int(x) for x in items_idxs]
        return items_idxs, items

    def get_similar_items(self, received_name: str = 'Bambi (1942)', amount: int = 5):
        item_id = self.__find_item_by_name(received_name)
        if self.items_similarity_matrix is None:
            self.items_similarity_matrix = self.__calculate_items_similarity_matrix(self.model.data.T)
        items_idxs, items = self.__find_similar(item_id, amount)
        return [items_idxs, items]

    def predict(self, items_ratings: list, M: int = 10):
        if len(items_ratings) != 2:
            raise Exception('Wrong input!')
        print('here')
        ratings = items_ratings[1]
        items_ids = items_ratings[0]
        print('here11')
        if not isinstance(items_ids[0], int):
            items_ids = [self.__find_item_by_name(x) for x in items_ids]
        data = [items_ids, ratings]
        print('here12')
        new_user_row = self.__init_new_row(data)
        print('new user row', new_user_row.shape)

        normalized_row, mean_user_rating, std_user_rating = self.__normalize_row(new_user_row)

        # print(new_user_row)

        new_user_to_users_similarity = cosine_similarity(normalized_row, self.user_item_matrix).T

        # u - степень похожести
        #

        print('here3')
        # prob_values = self.model.data * new_user_to_users_similarity

        most_similar_user_id = pd.DataFrame(data=cosine_similarity(normalized_row, self.model.data),
                                            columns=self.model.data.T.columns.values).idxmax(axis=1).max()

        # ранжируем по похожести
        # оцеку * на похожесть
        # оценки * на похожесть
        # оценки на всех

        # вектор на матрицу всех оценок (после норм оценок)->

        # print(most_similar_user_id)
        # numpy
        # labelencoder

        items_ids = [int(x) for x in
                     self.model.data.T[most_similar_user_id].sort_values(ascending=False)[:M].index.values.tolist()]
        items_ratings = self.model.data.T[most_similar_user_id].sort_values(ascending=False)[:M].values.tolist()
        items_names = self.__find_items_by_ids(items_ids).title.values.tolist()

        return [items_names, items_ratings]

    def __find_items_by_ids(self, ids: list):
        return self.items_matrix.loc[self.items_matrix['movie_id'].isin(ids), :]

    def __init_new_row(self, items_ratings: list):

        items_ids = items_ratings[0]
        ratings = items_ratings[1]

        # for (items_id, rating) in zip(items_ids, ratings):
        #     # if items_id in new_user_row.columns:
        #     new_user_row[f'{items_id}'] = rating

        values = np.empty((1, len(self.model.data.columns)))
        values.fill(np.nan)

        print('init row values', values.shape)
        print('init row values', len(self.model.data.columns))

        new_user_row = pd.DataFrame(data=values, columns=self.model.data.columns)
        print(new_user_row.shape)
        for (id, mark) in zip(items_ids, ratings):
            new_user_row[id] = mark
        print(new_user_row.shape)
        return new_user_row

    def __create_indexer_dict(self, items_ids: list):
        indexer = {}
        for i, val in enumerate(items_ids):
            indexer[val] = i
        return indexer

    def __sort_items_by_ids(self, items: pd.DataFrame, items_ids: list):
        indexer = self.__create_indexer_dict(items_ids)
        items = items.loc[items['movie_id'].isin(items_ids), :]
        items.loc[:, ['order']] = items['movie_id'].map(indexer)
        names = items.sort_values('order')['title'].values.tolist()
        return names

    def test(self):
        print('test')
        return 'test'


class CliWrapper(object):
    def __init__(self):
        self.options = RecSysOptions()
        self.recsys = RecSysMF(self.options)

    def train(self, dataset: str = None):
        self.recsys.train(dataset)

    def evaluate(self, dataset: str = None):
        self.recsys.train()
        rmse = self.recsys.evaluate(dataset)
        print(rmse)

    def predict(self, dataset: str = None):
        self.recsys.train()
        user_id = 50
        data = [[int(x) for x in self.recsys.model.data.T[user_id].index.values.tolist()],
                self.recsys.model.data.T[user_id].tolist()]
        output = self.recsys.predict(data)
        print(output)

    def surprise_train(self, dataset: str = None):
        self.recsys.surprise_train(dataset)

    def surprise_evaluate(self, dataset: str = None):
        self.recsys.surprise_train()
        rmse = self.recsys.surprise_evaluate(dataset)
        print(rmse)


if __name__ == "__main__":
    fire.Fire(CliWrapper)

