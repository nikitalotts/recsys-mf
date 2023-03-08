# -*- coding: utf-8 -*-

import os

import numpy as np
from flask import Flask, request, jsonify, make_response
from threading import Lock

# from main import predict, reload
import json

from pydotplus import basestring
from webapp.service import Service

LOCK = Lock()
app = Flask(__name__)
service = Service()

@app.route('/api/predict', methods=['POST'])
def predict():
    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy!'}), 403)

    with LOCK:
        request_data = request.get_json()
        if request_data is None:
            return make_response(jsonify({'message': 'Request body doesn\'t contain any data'}), 400)
        try:
            movie_names_ratings = request_data['movies_ratings']
            movie_names = movie_names_ratings[0]
            ratings = movie_names_ratings[1]

            if 'M' in request_data:
                m = request_data['M']

            if not isinstance(m, int) or (isinstance(m, int) and (m < 1 or m > 25)):
                m = 20

            if not isinstance(movie_names_ratings, list):
                return make_response(jsonify({'message': 'Given values are not double lists'}), 400)

            if len(movie_names_ratings) != 2:
                return make_response(jsonify({'message': 'Expected only two array'}), 400)

            if len(movie_names) != len(ratings):
                return make_response(jsonify({'message': 'Lists have different sizes'}), 400)

            if not is_list_of_strings(movie_names):
                return make_response(jsonify({'message': 'Movies names list consist not only of strings'}), 400)

            if not is_list_of_ints(ratings):
                return make_response(jsonify({'message': 'Movies names list consist not only of ints'}), 400)

        except KeyError:
            return make_response(jsonify({'message': 'No \'movie_names_ratings\' key in JSON data'}), 400)
        except json.decoder.JSONDecodeError:
            return make_response(jsonify({'message': 'Invalid JSON data'}), 400)

        try:
            result = service.predict([movie_names, ratings], m)
            return make_response(jsonify(result), 200)
        except Exception as e:
            print(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)



#

@app.route('/api/log', methods=['GET'])
def log():
    # try exept
    # with open("app.log", "r") as log_f:
    #     logs_tail = log_f.readlines()[-20:]

    print(service.hello('hello hui'))

    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)

    with LOCK:
        try:
            # run service func
            return make_response(jsonify({'Last 20s app.log rows': 'qwdqwd'}))
        except:
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/info', methods=['GET'])
def info():
    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)

    with LOCK:
        try:
            # run service func
            return make_response(jsonify({'message': "Will be here soon :-)"}))
        except:
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/reload', methods=['POST'])
def reload():
    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)

    with LOCK:
        try:
            service.reload()
            return make_response(jsonify({'Result': "Model successfully reloaded!"}))
        except:
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/similar', methods=['POST'])
def similar():
    if LOCK.locked():
        return make_response(jsonify({'error': 'Processing in progress!'}), 403)

    with LOCK:
        request_data = request.get_json()
        if request_data is None:
            return make_response(jsonify({'message': 'Request body doesn\'t contain JSON data'}), 400)
        try:
            n = 5
            movie_name = request_data['movie_name']

            if 'N' in request_data:
                n = request_data['N']
                if not isinstance(n, int) or (isinstance(n, int) and n > 50):
                    n = 5

            if not isinstance(movie_name, str) or movie_name == '':
                return make_response(jsonify({'message': 'Wrong \'movie_name\' value'}), 400)

        except KeyError:
            return make_response(jsonify({'message': 'No \'movie_name\' key in JSON data'}), 400)
        except json.decoder.JSONDecodeError:
            return make_response(jsonify({'message': 'Invalid JSON data'}), 400)

        try:
            output = service.similair(movie_name, n)
            return make_response(jsonify(output), 200)
        except:
            return make_response(jsonify({'error': 'Something went wrong'}), 500)

@app.route('/api/test/', methods=['GET'])
def test():
    print(request.args.getlist('movies_ratings'))


# @app.route('/api/old_predict', methods=['GET'])
# def process():
#     user_id = request.args.get('user_id', default=100, type=int)
#     M_items_recommend = request.args.get('M', default=20, type=int)
#
#     if LOCK.locked():
#         return make_response(jsonify({'error': 'Processing in progress!'}), 403)
#
#     with LOCK:
#         try:
#             movies, ratings = ['#movies'], ['ratings']
#         except Exception as e:
#             print(e)
#             return make_response(jsonify({'error': f'{e}'}), 500)
#         else:
#             return make_response(jsonify({'result': f"Movies: {movies}  With Ratings: {ratings}"}))


def is_list_of_strings(lst):
    return bool(lst) and isinstance(lst, list) and all(isinstance(elem, basestring) for elem in lst)


def is_list_of_ints(lst):
    return all(isinstance(x, int) for x in lst)
