from __future__ import absolute_import
import fire
import pytz
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
        self.creation_time = datetime.now(pytz.timezone("Asia/Barnaul")).strftime("%m-%d-%Y:::%H:%M:%S.%f:::UTC%Z")
        # self.creation_date = str(datetime.now())
        logging.info('RecSysMF instance successfully inited')

    # def warmup(self, model_name: str='model', model_extension: str='csv'):
    def warmup(self, model_type: str = 'SVD'):
        logging.info(f'started warmup method, model_type:{model_type}, model_data_path:{self.options.model_data_path}')
        if model_type == 'SVD':
            self.model = SvdModel()
        else:
            logger.error(f'NameError: Invalid model type: {model_type}')
            raise NameError('tried to load invalid model type: {model_type}')

        if self.is_model_exists(self.options.model_data_path):
            self.model.load_data(self.options)
        else:
            logger.warning(f'model doesn\'t exist: {self.options.model_data_path}')

        self.users_matrix, self.n_users = self.load_users_data(self.options.users_data_path)
        self.items_matrix, self.n_items = self.load_items_data(self.options.items_data_path)

        self.users_matrix = self.proceed_users(self.users_matrix)
        self.items_matrix = self.proceed_items(self.items_matrix)
        # logger.info(f"Model: {self.options.model} successfully loaded: {datetime.now()}")

        # renew results
        self.options.current_accuracy = 'Model haven\'t been evaluated yet'
        self.options.datetime_accuracy_test = 'Model haven\'t been evaluated yet'

    def train(self, train_data_path: str = None):
        logger.info(f'started train method, {train_data_path}')
        self.warmup()

        if train_data_path is None:
            train_data_path = self.options.train_data_path
            logger.info(f'given train_data_path is none, train_data_path changed to options.train_data_path: {self.options.train_data_path}')

        self.ratings_train = self.load_ratings(train_data_path)
        self.user_item_matrix = self.create_user_item_matrix(self.users_matrix, self.items_matrix, self.ratings_train)

        self.user_item_matrix, self.mean_user_rating, self.std_user_rating = self.normalize_matrix(
            self.user_item_matrix)
        self.model.fit(self.user_item_matrix, self.options.n_vectors, self.mean_user_rating,
                       self.std_user_rating)  # , self.std_user_rating)
        self.model.save(self.options.model_name, self.options)
        self.trained = True
        print('trained')
        return

    def is_model_exists(self, model_data_path: str):
        """
        все модели сохраняются в моделс сторе
        """
        return os.path.exists(model_data_path)

    def create_user_item_matrix(self, users: pd.DataFrame, items: pd.DataFrame, ratings: pd.DataFrame):
        logger.info(f'started create_user_item_matrix method')
        user_item_rating_dataframe = self.create_user_item_rating_dataframe(users, items, ratings)
        matrix = user_item_rating_dataframe.pivot(index='user_id', columns='movie_id', values='rating')
        logger.info(f'create_user_item_matrix method successfully executed')
        return matrix

    def create_user_item_rating_dataframe(self, users: pd.DataFrame, items: pd.DataFrame, ratings: pd.DataFrame):
        logger.info('started create_user_item_rating_dataframe method')
        dataframe = pd.merge(ratings, items, on='movie_id', how='left').merge(users, on='user_id', how='left')
        logger.info(f'create_user_item_rating_dataframe method successfully executed')
        return dataframe

    def load_model_data(self, model_path: str):
        model_path_split = os.path.splitext(model_path)
        self.options.model_name = model_path_split[0]
        self.options.model_extention = model_path_split[1][1:]

        if self.options.model_extention == 'csv':
            self.model = pd.read_csv(self.options.model_store + f'{model_path}', encoding=self.options.encoding)
        else:
            raise Exception("Wrong model extension")
        return

    def load_users_data(self, users_data: str):
        logger.info('started load_users_data method')
        users = pd.read_csv(users_data, names=['user_id', 'gender', 'age', 'occupation', 'zip-code'],
                            sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                            encoding=self.options.encoding)
        n_users = users['user_id'].nunique()

        logger.info('load_users_data method successfully executed')
        return users, n_users

    # вынести
    def load_items_data(self, items_data: str):
        items = pd.read_csv(items_data, names=['movie_id', 'title', 'genres'],
                            sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                            encoding=self.options.encoding)
        n_items = items['movie_id'].nunique()

        return items, n_items

    # def load_ratings(self, ratings_data_path: str, is_train: bool=True):
    def load_ratings(self, ratings_data_path: str):
        logger.info(f'started load_ratings method, {ratings_data_path}')
        ratings = pd.read_csv(ratings_data_path, names=['user_id', 'movie_id', 'rating', 'date'],
                              sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                              encoding=self.options.encoding)
        # if is_train:
        #     self.ratings_train = ratings
        # else:
        #     self.ratings_test = ratings
        logger.info(f'load_ratings method successfully inited')
        return ratings

    def proceed_items(self, items_matrix: pd.DataFrame):
        logger.info('started proceed_items method')
        items_matrix['release_year'] = items_matrix['title'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)
        logger.info('proceed_items method successfully executed')
        return items_matrix

    def proceed_users(self, users_matrix: pd.DataFrame):
        logger.info('started proceed_users method')
        logger.info('proceed_users method successfully executed')
        return users_matrix

    def normalize_matrix(self, matrix: pd.DataFrame):
        logger.info('started normalize_matrix method')
        # mean_user_rating = np.mean(matrix.values, axis=1).reshape(-1, 1)
        mean_user_rating = np.nanmean(matrix.values, axis=1).reshape(-1, 1)
        std_user_rating = np.nanstd(matrix.values, axis=1).reshape(-1, 1)
        # std_user_rating = np.std(matrix.values, axis=1).reshape(-1, 1)
        # matrix_normalized_values = (matrix.values - mean_user_rating) / mean_user_rating
        matrix_normalized_values = (matrix.values - mean_user_rating) / std_user_rating
        matrix = pd.DataFrame(data=matrix_normalized_values, index=matrix.index, columns=matrix.columns).fillna(0)
        mean_user_rating[np.isnan(mean_user_rating)] = 0
        std_user_rating[np.isnan(std_user_rating)] = 0

        logger.info('normalize_matrix method successfully executed')
        return matrix, mean_user_rating, std_user_rating

    def normalize_row(self, row: pd.DataFrame):
        logger.info('started normalize_row method')
        mean_user_rating = np.nanmean(row.values, axis=1).reshape(-1, 1)
        std_user_rating = np.nanstd(row.values, axis=1).reshape(-1, 1)
        row_normalized_values = (row.values - mean_user_rating) / std_user_rating
        row = pd.DataFrame(data=row_normalized_values, index=row.index, columns=row.columns).fillna(0)
        logger.info('normalize_row method successfully executed')
        return row, mean_user_rating, std_user_rating

    def get_movies_ids(self, predictions: pd.DataFrame):
        logger.info('started get_movies_ids method')
        ids = predictions.columns.values
        logger.info('get_movies_ids method successfully executed')
        return [int(x) for x in ids]

    def evaluate(self, test_data_path: str = None):
        logger.info('started evaluate method')
        if self.trained == False:
            logger.error('evaluate method not executed as model not train')
            raise Exception('model not trained')

        self.warmup()
        if test_data_path == None:
            test_data_path = self.options.test_data_path

        # print('test_Data_path', self.options.test_data_path)

        self.ratings_test = self.load_ratings(test_data_path)

        # print('self.ratings_test', self.ratings_test)

        test_dataset = self.create_user_item_rating_dataframe(self.users_matrix, self.items_matrix, self.ratings_test)

        # print('self.calculate_rmse(test_dataset, self.model.data)')
        # print(self.model.data, test_dataset)

        rmse = self.calculate_rmse(test_dataset, self.model.data)
        # rmse2 = self.calc_rmse2(self.model.data)

        print(f'RMSE: {rmse}')
        logger.info(f'evaluated with RMSE:{rmse}')
        # print(f'RMSE: {rmse2}')
        self.set_evaluate_results(rmse)
        logger.info('evaluate method successfully executed')
        return rmse

    def calculate_rmse(self, dataset: pd.DataFrame, preds: pd.DataFrame):
        logger.info('started calculate_rmse method')
        real_marks = []
        predictions = []
        for index, row in dataset.iterrows():
            user_id = row['user_id'] - 1
            movie_id = row['movie_id']
            rating = row['rating']
            # if movie_id in preds.columns:
            real_marks.append(rating)
            predictions.append(preds[movie_id][user_id])

        logger.info('calculate_rmse method successfully executed')
        return mean_squared_error(real_marks, predictions, squared=False)

    # def calc_rmse2(self, preds):
    #     test_dataset = self.create_user_item_matrix(self.users_matrix, self.items_matrix, self.ratings_test)
    #     print(test_dataset)
    #     rmse = np.nanmean((test_dataset.values - preds.values) ** 2)
    #     print('rmse2', rmse)
    #     return rmse

    #### surprise ####
    def surprise_train(self, train_data_path: str = None):
        logger.info('started surprise_train method')
        if train_data_path == None:
            train_data_path = self.options.train_data_path

        self.ratings_train = self.load_ratings(train_data_path)

        dataset = self.surprise_get_dataset(self.ratings_train)
        self.surprise_fit_model(dataset)
        self.trained = True
        print('trained')
        logger.info('surprise_train method successfully executed')
        return

    def surprise_get_dataset(self, ratings: pd.DataFrame):
        return Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader=Reader(rating_scale=(1, 5)))

    def surprise_fit_model(self, dataset: surprise.Dataset):
        logger.info('stared surprise_fit_model method')
        self.model = SVD(n_factors=50)
        self.model.fit(dataset.build_full_trainset())
        logger.info('surprise_fit_model method successfully executed')

    def surprise_make_predictions(self, dataset: surprise.Dataset):
        logger.info('stared surprise_make_predictions method')
        real_marks = []
        predictions = []
        for row in dataset.build_full_trainset().build_testset():
            real_marks.append(row[2])
            predictions.append(self.model.predict(row[0], row[1]).est)

        logger.info('surprise_make_predictions method successfully executed')
        return np.array(real_marks), np.array(predictions)

    def surprise_calculate_rmse(self, real: np.matrix, pred: np.matrix):
        return mean_squared_error(real, pred, squared=False)

    def surprise_evaluate(self, test_data_path: str = None):
        logger.info('stared surprise_evaluate method')
        if self.trained == False:
            logger.error('surprise_evaluate method not executed as model not train')
            raise Exception('surprise_evaluate: model not trained')

        if test_data_path == None:
            test_data_path = self.options.test_data_path
        self.ratings_test = self.load_ratings(test_data_path)
        dataset = self.surprise_get_dataset(self.ratings_test)
        real_marks, predictions = self.surprise_make_predictions(dataset)
        rmse = self.surprise_calculate_rmse(real_marks, predictions)
        self.set_evaluate_results(rmse)
        logger.info(f'surprise: evaluated with RMSE:{rmse}')
        logger.info('surprise_evaluate method successfully executed')
        return rmse

    # query handlers

    def find_item_by_name(self, received_name: str):
        item_index = self.items_matrix['title'].apply(
            lambda title: levenshtein(re.sub(r' \([0-9]{4}\)', '', title.lower()), received_name.lower())).idxmin()
        item_id = self.items_matrix.loc[item_index]['movie_id']
        return item_id

    def calculate_items_similarity_matrix(self, items_matrix: pd.DataFrame):
        # t тк вектор - фильм
        similarity_matrix = cosine_similarity(items_matrix, items_matrix)
        # нуля чтоб сам себя не рекоммендовал

        np.fill_diagonal(similarity_matrix, 0)

        similarity_df = pd.DataFrame(similarity_matrix, self.model.data.columns, self.model.data.columns)
        return similarity_df

    def find_similar(self, movie_id: int, n: int = 5):
        items_idxs = np.array(self.items_similarity_matrix[movie_id].sort_values(ascending=False)[:n].index.values,
                              dtype=int).tolist()

        items = self.sort_items_by_ids(self.items_matrix, items_idxs)

        items_idxs = [int(x) for x in items_idxs]
        return items_idxs, items

    def get_similar_items(self, received_name: str = 'Bambi (1942)', amount: int = 5):
        item_id = self.find_item_by_name(received_name)
        if self.items_similarity_matrix is None:
            self.items_similarity_matrix = self.calculate_items_similarity_matrix(self.model.data.T)
        items_idxs, items = self.find_similar(item_id, amount)
        return [items_idxs, items]

    def predict(self, items_ratings: list, M: int = 10):
        logger.info('started predict method')
        if len(items_ratings) != 2:
            logger.error('Wrong input: array dim must equals 2')
            raise ValueError('Wrong input: array dim is not equals 2')

        # print('here')
        ratings = items_ratings[1]
        items_ids = items_ratings[0]
        # print('here11')
        if not isinstance(items_ids[0], int):
            items_ids = [self.find_item_by_name(x) for x in items_ids]
        data = [items_ids, ratings]
        # print('here12')
        new_user_row = self.init_new_row(data)
        # print('new user row', new_user_row.shape)

        normalized_row, mean_user_rating, std_user_rating = self.normalize_row(new_user_row)
        logger.info('normalized_row row in predict method')

        # print(new_user_row)

        weights = cosine_similarity(normalized_row.values * 1000, self.user_item_matrix * 1000).T
        logger.info('calcualte similarity matrix in predict')

        # gt0 = (weights > 0).sum()

        # print('gt00', gt0)

        weights = np.broadcast_arrays(weights, self.user_item_matrix)[0]

        output = np.average(self.user_item_matrix, axis=0, weights=weights)


        # print('just average', output)

        marks = (output * std_user_rating + mean_user_rating)[0]

        logger.info('calculate marks in predict method')

        # print('marks', marks, marks.shape)
        #
        # print('output shape', output.shape)
        #
        # print('data shape norm', self.model.data)
        # print('data shape T', self.model.data.T)

        # print('output shape', marks.shape)
        #
        # if 1 not in marks.shape:
        #     print('EROEROEROEEO')

        # для всех можно
        best_movies_ids = np.argsort(-marks.reshape(1, -1), axis=1)[:, :M]

        # print('best movies', best_movies_ids, type(best_movies_ids))

        best_movies_cols_ids = self.model.data.columns.values[best_movies_ids][0]

        # print('best_movies_cols_ids', best_movies_cols_ids, type(best_movies_cols_ids), best_movies_cols_ids[0])
        # print(best_movies_ids.shape)

        films = self.find_items_by_ids(best_movies_cols_ids)

        # print('films', films)

        films.sort_values('movie_id', ascending=False)['title'].values.tolist()

        movies_names = self.sort_items_by_ids(films, best_movies_cols_ids)
        logger.info('recieved best movies names in predict method')

        # print('asd', movies_names)

        marks.sort()

        # print('marks', marks)

        best_movies_marks = marks[::-1][:M]

        # print('best_movies_marks', best_movies_marks)
        #
        # print('result', movies_names, best_movies_marks)

        best_movies_marks[best_movies_marks > 5] = 5
        best_movies_marks[best_movies_marks < 0] = 0

        # most_similar_user_id = pd.DataFrame(data=cosine_similarity(normalized_row, self.model.data),
        #                                     columns=self.model.data.T.columns.values).idxmax(axis=1).max()

        # ранжируем по похожести
        # оцеку * на похожесть
        # оценки * на похожесть
        # оценки на всех

        # вектор на матрицу всех оценок (после норм оценок)->

        # print(most_similar_user_id)
        # numpy
        # labelencoder

        # items_ids = [int(x) for x in
        #              self.model.data.T[most_similar_user_id].sort_values(ascending=False)[:M].index.values.tolist()]
        # items_ratings = self.model.data.T[most_similar_user_id].sort_values(ascending=False)[:M].values.tolist()
        # items_names = self.find_items_by_ids(items_ids).title.values.tolist()

        logger.info('predict method successfully executed')
        return [movies_names, best_movies_marks.tolist()]

    def predict_dataset(self, dataset_path: str, m: int = 5):
        logger.info('started predict_dataset method')
        if dataset_path is None:
            dataset_path = self.options.test_data_path

        self.ratings_test = self.load_ratings(self.options.test_data_path)
        test_matrix = self.create_user_item_matrix(self.users_matrix, self.items_matrix, self.ratings_test)

        test_matrix = self.fill_missed_values(test_matrix)

        top_items_names, top_items_ratings = self.find_top_items_for_users(test_matrix, m)

        result = self.generate_dataframe(top_items_names, top_items_ratings, m)

        name = f'prediction_for_top_{m}_movies'
        result.to_csv(name)
        logger.info('predict_dataset method successfully executed')
        return str(os.getcwd() + name)

    def generate_dataframe(self, items_names: list, items_ratings: list, m: int = 5):
        columns = self.generate_new_columns(m)

        data = []
        for name, mark in zip(items_names, items_ratings):
            row_data = []
            for i in range(m):
                row_data.append(name[i])
                row_data.append(mark[i])
            data.append(row_data)

        res_df = pd.DataFrame(data=data, columns=columns)
        res_df.index = res_df.index + 1
        logger.info('generate_dataframe method successfully executed')
        return res_df

    def find_top_items_for_users(self, test_matrix: pd.DataFrame, m: int = 5):
        normalized_matrix, mean_user_rating, std_user_rating = self.normalize_matrix(test_matrix)

        user_to_user_similarity = cosine_similarity(normalized_matrix.values, self.user_item_matrix).T

        top_items_names = []
        top_items_ratings = []

        for (asd, user_data) in enumerate(zip(user_to_user_similarity, std_user_rating, mean_user_rating)):

            if asd % 100 == 0 and asd > 0:
                print(f'epoch {asd}')
                break

            output = np.average(self.user_item_matrix, axis=0,
                                weights=np.broadcast_arrays(user_data[0].reshape(-1, 1), self.user_item_matrix)[0])
            output = output * user_data[1] + user_data[2]

            best_movies_ids = np.argsort(-output.reshape(1, -1), axis=1)[:, :m]
            best_movies_cols_ids = self.model.data.columns.values[best_movies_ids][0]

            items = self.find_items_by_ids(best_movies_cols_ids)
            items.sort_values('movie_id', ascending=False)['title'].values.tolist()
            movies_names = self.sort_items_by_ids(items, best_movies_cols_ids)

            best_movies_marks = np.sort(output)[::-1][:m]
            best_movies_marks[best_movies_marks > 5] = 5
            best_movies_marks[best_movies_marks < 0] = 0

            top_items_names.append(movies_names)
            top_items_ratings.append(best_movies_marks)

        logger.info('find_top_items_for_users method successfully executed')
        return top_items_names, top_items_ratings


    def generate_new_columns(self, m):
        cols = []
        for i in range(1, m + 1):
            cols.append(f'movie_name_{i}')
            cols.append(f'mark_{i}')

        return cols

    def fill_missed_values(self, test_matrix: pd.DataFrame):
        # movies
        train_columns = self.model.data.columns.values
        train_index = self.model.data.index.values

        # users
        test_columns = test_matrix.columns.values
        test_index = test_matrix.index.values - 1

        movies_lack_in_test = set(train_columns) - set(test_columns)
        users_lack_in_test = set(train_index) - set(test_index)

        new_columns = np.append(test_columns, list(movies_lack_in_test))
        new_index = np.append(test_index, list(users_lack_in_test))

        new_data = np.empty((len(users_lack_in_test), test_matrix.shape[1]))
        new_data[:] = np.nan

        matrix_filled_users = np.vstack([test_matrix.values, new_data])

        new_data = np.empty((matrix_filled_users.shape[0], len(movies_lack_in_test)))
        new_data[:] = np.nan

        new_data = np.hstack([matrix_filled_users, new_data])

        filled_matrix = pd.DataFrame(data=new_data, columns=new_columns, index=new_index)
        filled_matrix = filled_matrix.sort_index()
        filled_matrix = filled_matrix.reindex(sorted(filled_matrix.columns), axis=1)

        logger.info('fill_missed_values method successfully executed')
        return filled_matrix

    def find_items_by_ids(self, ids: list):
        return self.items_matrix.loc[self.items_matrix['movie_id'].isin(ids), :]

    def init_new_row(self, items_ratings: list):

        items_ids = items_ratings[0]
        ratings = items_ratings[1]

        # for (items_id, rating) in zip(items_ids, ratings):
        #     # if items_id in new_user_row.columns:
        #     new_user_row[f'{items_id}'] = rating

        values = np.empty((1, len(self.model.data.columns)))
        values.fill(np.nan)

        new_user_row = pd.DataFrame(data=values, columns=self.model.data.columns)
        for (id, mark) in zip(items_ids, ratings):
            new_user_row[id] = mark

        logger.info('init_new_row method successfully executed')
        return new_user_row

    def create_indexer_dict(self, items_ids: list):
        indexer = {}
        for i, val in enumerate(items_ids):
            indexer[val] = i

        logger.info('create_indexer_dict method successfully executed')
        return indexer

    def sort_items_by_ids(self, items: pd.DataFrame, items_ids: list):
        indexer = self.create_indexer_dict(items_ids)
        items = items.loc[items['movie_id'].isin(items_ids), :]
        items.loc[:, ['order']] = items['movie_id'].map(indexer)
        names = items.sort_values('order')['title'].values.tolist()
        logger.info('sort_items_by_ids method successfully executed')
        return names

    def set_evaluate_results(self, rmse: float):
        self.options.current_accuracy = rmse
        self.options.datetime_accuracy_test = datetime.now(pytz.timezone("Asia/Barnaul")).strftime("%m-%d-%Y:::%H:%M:%S.%f:::UTC%Z")

    def get_info(self):
        info_dict = {
            'accuracy(rmse)' : self.options.current_accuracy,
            'time': self.options.datetime_accuracy_test,
            'current_user': self.options.credentials,
            'model_author': self.options.author,
            'docker_built_time': self.creation_time
        }

        return info_dict


class CliWrapper(object):
    def __init__(self):
        logger.info('started CliWrapper instance initialization')
        self.options = RecSysOptions()
        self.recsys = RecSysMF(self.options)
        logger.info('CliWrapper instance successfully inited')

    def train(self, dataset: str = None):
        self.recsys.train(dataset)
        logger.info('train method successfully executed')

    def evaluate(self, dataset: str = None):
        self.recsys.train()
        rmse = self.recsys.evaluate(dataset)
        logger.info('evaluate method successfully executed')

    def predict(self, dataset: str = None, amount: int = 5):
        self.recsys.train()
        output = self.recsys.predict_dataset(dataset, amount)
        print(output, 'predicted')
        logger.info('predict method successfully executed')

    def surprise_train(self, dataset: str = None):
        self.recsys.surprise_train(dataset)
        logger.info('surprise_train method successfully executed')

    def surprise_evaluate(self, dataset: str = None):
        self.recsys.surprise_train()
        rmse = self.recsys.surprise_evaluate(dataset)
        print(rmse)
        logger.info('surprise_evaluate method successfully executed')


if __name__ == "__main__":
    fire.Fire(CliWrapper)
    logger.info('fire\' CliWrapper is running')
