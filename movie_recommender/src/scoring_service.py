import logging
import sys

import flask
from flask import Flask, request
from flask_cors import CORS

from movie_recommender.src.model import MovieRecommender
from movie_recommender.src import config


# Python2.x: Make default encoding as UTF-8
from util.data_store.local_filesystem import LocalFileSystem

if sys.version_info.major == 2:
    reload(sys)
    sys.setdefaultencoding('UTF8')


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
app = Flask(__name__)
CORS(app)

global movie_reco_model


@app.before_first_request
def load_model():
    model_data_store = LocalFileSystem(src_dir="/movie_recommender/data")
    assert (model_data_store is not None)

    app.movie_reco_model = MovieRecommender.load_from_data_store(data_store=model_data_store)
    assert app.movie_reco_model is not None

    app.logger.info("movie recommendation model got loaded successfully!")


@app.route('/')
def heart_beat():
    return flask.jsonify({"status": "ok"})


@app.route('/api/v1/recommend_movies')
def find_user_category():
    userid = request.args.get('userid')
    response = app.movie_reco_model.recommend_movies(user_id=int(userid))
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run()
