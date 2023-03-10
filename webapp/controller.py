# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from flask import Flask, request, jsonify, make_response
from threading import Lock

import json

from pydotplus import basestring
from webapp.service import Service

LOCK = Lock()
app = Flask(__name__)
service = Service()
logger = logging.getLogger(__name__)


@app.route('/api/predict', methods=['POST'])
def predict():
    logger.info('reached /api/predict/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy'}), 403)

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
        except KeyError as e:
            logger.error(e)
            return make_response(jsonify({'message': 'No \'movie_names_ratings\' key in JSON data'}), 400)
        except json.decoder.JSONDecodeError as e:
            logger.error(e)
            return make_response(jsonify({'message': 'Invalid JSON data'}), 400)

        try:
            result = service.predict([movie_names, ratings], m)
            logger.info('result successfully predicted')
            return make_response(jsonify(result), 200)
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/log', methods=['GET'])
def log():
    # try exept
    # with open("app.log", "r") as log_f:
    #     logs_tail = log_f.readlines()[-20:]

    logger.info('reached /api/log/ endpoint')

    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy'}), 403)
    with LOCK:
        try:
            output = service.log()
            logger.info('logs successfully received')
            return make_response(jsonify({'last 20 rows of logs': output}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/info', methods=['GET'])
def info():
    logger.info('reached /api/info/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy'}), 403)
    with LOCK:
        try:
            info = service.info()
            logger.info('docker info successfully recieved')
            return make_response(jsonify({'message': info}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/reload', methods=['POST'])
def reload():
    logger.info('reached /api/reload/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'server is busy'}), 403)
    with LOCK:
        try:
            service.reload()
            logger.info('model successfully reloaded')
            return make_response(jsonify({'Result': "model successfully reloaded"}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    logger.info('reached /api/evaluate/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'server is busy'}), 403)
    with LOCK:
        try:
            service.evaluate()
            logger.info('model successfully evaluated')
            return make_response(jsonify({'Result': "model successfully evaluated"}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/surprise_evaluate', methods=['POST'])
def surprise_evaluate():
    logger.info('reached /api/surprise_evaluate/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'server is busy'}), 403)
    with LOCK:
        try:
            service.surprise_evaluate()
            logger.info('model successfully evaluated, surprise')
            return make_response(jsonify({'Result': "model successfully evaluated surprise"}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/api/similar', methods=['POST'])
def similar():
    logger.info('reached /api/similar/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy'}), 403)
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
        except KeyError as e:
            logger.error(e)
            return make_response(jsonify({'message': 'No \'movie_name\' key in JSON data'}), 400)
        except json.decoder.JSONDecodeError as e:
            logger.error(e)
            return make_response(jsonify({'message': 'Invalid JSON data'}), 400)
        try:
            output = service.similar(movie_name, n)
            logger.info('successfully recieved similair results')
            return make_response(jsonify(output), 200)
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


def is_list_of_strings(lst):
    return bool(lst) and isinstance(lst, list) and all(isinstance(elem, basestring) for elem in lst)


def is_list_of_ints(lst):
    return all(isinstance(x, int) for x in lst)
