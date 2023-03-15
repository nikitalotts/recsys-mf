
# Recommender System Based on Matrix Factorization 

This project represents the implementation of a recommendation system based on matrix factorization performed on the 5th module of the 3rd year of HITS, Tomsk State University, Tomsk.

##  Installation

### Within Docker
If you want to run project in Docker - follow these instructions:

 ```
1. Enter into project's folder
    $ cd /path/to/recsys-mf/
2. Run docker compose*
    $ docker-compose up --build
   
   Or if you want to run in detach mode:
   $ docker-compose up --detach --build
 ```

 Now server is running and listening on `127.0.0.1:5000`

 ```
 *If you have troubles with docker-compose while loading metadate,
 try run this before starting composing:
    $ docker pull python:3.10.6-buster
 ```

 ### Without Docker

If you want to run project without Docker:
 ```
1. Copy(or move) data and logs folders from basedir to ./recsys (inside docker volumes do that)
2. In recsys-mf/recsys/ create credentials.txt and write two lines:     
    a) current date 
    b) "@nikitalotts" (there are steps in Dockerfile that do that automatically but now we need to immitate them)
3. run recsys-mf/main.py
 ```
  Now server is running on the same address

### CLI
There four CLI commands that system suppurt(you can use them within docker container or in terminal):
1. train (standart model):

```$ python path/to/model.py train --dataset='path/to/data/train/ratings_train.dat'(by default, '../data/train/ratings_train.dat')```

2. evaluate (standart model):

```$ python path/to/model.py evaluate --dataset='path/to/data/test/ratings_test.dat' (by default, '../data/test/ratings_test.dat')```

3. train (sklearn-surprise based model):

```$ python path/to/model.py surprise_train --dataset='path/to/data/train/ratings_train.dat'(by default, '../data/train/ratings_train.dat')```

4. evaluate (sklearn-surprise based model):

```$ python path/to/model.py surprise_evaluate --dataset='path/to/data/test/ratings_test.dat' (by default, '../data/test/ratings_test.dat')```

5. predict:

```$ python path/to/model.py predict --dataset='path/to/data/test/ratings_test.dat' (by default, '../data/test/ratings_test.dat')```


### Available API Requests
Endpoints:
- `/api/predict`. Recieves list `[[movie_name_1, movie_name_2, .., movie_name_N ], [rating_1, rating_2, .., rating_N]]` and returns TOP M (default 20, also a parameter) recommended movies with corresponding estimated rating. Sort descending. `[[movie_name_1, movie_name_2, .., movie_name_M], [rating_1, rating_2, .., rating_M]]`
- `/api/log`. Last 20 rows of log.
- `/api/info`. Service Information: Your Credentials, Date and time of the build of the Docker image, Date, time and metrics of the training of the currently deployed model.
- `/api/reload`. Reload the model.
- `/api/similar`. returns list of similar movies `{"movie_name": "Lord of the Rings"}`  
- `api/surprise_evaluate`. evaluate sklearn-surprise based model and renew best accuracy.

### Note
By default server is in deployment mode, if you want to change it to development mode, in `/recsys-mf/main.py` change `mode` variable's value to `dev` (return `prod` if you want to return deployment mode again)

### Sources
* [Рекомендательная система на основе SVD разложения матриц](https://www.youtube.com/watch?v=DM_lGbfoIYM)
* [Recommender System — Matrix Factorization](https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b)
* [How does Netflix recommend movies? Matrix Factorization](https://www.youtube.com/watch?v=ZspR5PZemcs)

### Author
*Nikita Lotts, 3rd grade student in Tomsk State University (Tomsk)*