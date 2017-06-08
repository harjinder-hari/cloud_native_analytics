from movie_recommender.src import config
from movie_recommender.src.model import MovieRecommender
from util.data_store.local_filesystem import LocalFileSystem
from util.data_store.s3_data_store import S3DataStore


def test_movie_recommender():
    movie_reco = MovieRecommender.train(src_url="data/sample_movielens_ratings.txt")
    assert movie_reco is not None

    movie_reco.save(target_url="/tmp")

    movie_reco = MovieRecommender.load(src_url="/tmp")
    assert movie_reco is not None

    reco = movie_reco.recommend_movies(user_id=25)
    assert reco is not None
    assert len(reco.items()) == 10


def test_movie_recommender_with_local_data_store():
    movie_reco = MovieRecommender.train(src_url="data/sample_movielens_ratings.txt")
    assert movie_reco is not None

    model_data_store = LocalFileSystem(src_dir="/tmp")
    movie_reco.save_to_data_store(data_store=model_data_store)

    movie_reco = MovieRecommender.load_from_data_store(data_store=model_data_store)
    assert movie_reco is not None

    reco = movie_reco.recommend_movies(user_id=25)
    assert reco is not None
    assert len(reco.items()) == 10


def test_movie_recommender_with_s3_data_store():
    movie_reco = MovieRecommender.train(src_url="data/sample_movielens_ratings.txt")
    assert movie_reco is not None

    model_data_store = S3DataStore(src_bucket_name=config.AWS_BUCKET,
                                   access_key=config.AWS_ACCESS_KEY_ID,
                                   secret_key=config.AWS_SECRET_ACCESS_KEY)
    assert (model_data_store is not None)

    movie_reco.save_to_data_store(data_store=model_data_store)

    movie_reco = MovieRecommender.load_from_data_store(data_store=model_data_store)
    assert movie_reco is not None

    reco = movie_reco.recommend_movies(user_id=25)
    assert reco is not None
    assert len(reco.items()) == 10
