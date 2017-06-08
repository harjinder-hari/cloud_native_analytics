import logging

from movie_recommender.src import config
from movie_recommender.src.model import MovieRecommender
from util.data_store.s3_data_store import S3DataStore

logging.basicConfig(filename=config.LOGFILE_PATH, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train_movie_recommender():
    movie_reco = MovieRecommender.train(src_url=config.TRAINING_DATA)

    model_data_store = S3DataStore(src_bucket_name=config.AWS_BUCKET,
                                   access_key=config.AWS_ACCESS_KEY_ID,
                                   secret_key=config.AWS_SECRET_ACCESS_KEY)
    assert (model_data_store is not None)

    movie_reco.save_to_data_store(data_store=model_data_store)


if __name__ == '__main__':
    train_movie_recommender()
