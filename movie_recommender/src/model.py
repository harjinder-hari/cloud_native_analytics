# imports ...

import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql import SparkSession

import config


class MovieRecommender:
    """
    A movie recommender system based on matrix factorization model.
    """

    def __init__(self):
        """
        Instantiates a movie recommender system.
        """
        self.dict_user = None
        self.dict_prod = None

    @classmethod
    def train(cls, src_url):
        """
        Train a matrix factorization model for movie recommendations.

        :param src_url: URL of source data for training the model
        :return: MovieRecommender object.
        """
        if config.LOCAL_RUN:
            os.environ["SPARK_HOME"] = config.LOCAL_SPARK_HOME  # Needed only when running local spark

        # Initialize spark context
        spark = SparkSession.builder.getOrCreate()

        lines = spark.read.text(src_url).rdd
        parts = lines.map(lambda row: row.value.split("::"))
        ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                             rating=float(p[2]), timestamp=long(p[3])))
        ratings = spark.createDataFrame(ratingsRDD)
        (training, test) = ratings.randomSplit([0.8, 0.2])

        # Build the recommendation model using ALS on the training data
        als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
        model = als.fit(training)
        tmp_url = "/tmp/als"
        model.write().overwrite().save(tmp_url)

        # Read the ALS model and extract user and product features
        user_path = join(tmp_url, "userFactors")
        product_path = join(tmp_url, "itemFactors")

        user_features_df = spark.read.format("parquet").load(user_path)
        product_features_df = spark.read.format("parquet").load(product_path)

        pd_user_df = user_features_df.toPandas().set_index('id')
        pd_prod_df = product_features_df.toPandas().set_index('id')

        movie_reco = MovieRecommender()
        movie_reco.dict_user = pd_user_df.to_dict()
        movie_reco.dict_prod = pd_prod_df.to_dict()

        return movie_reco

    def save(self, target_url):
        """
        Save the given movie recommendation model.

        :param target_url: URL where to save the movie recommendation model.
        :return: None
        """
        pickle.dump(self.dict_user, open(target_url + "/user.pkl", "wb"))
        pickle.dump(self.dict_prod, open(target_url + "/prod.pkl", "wb"))

    def save_to_data_store(self, data_store):
        """
        Save the given movie recommendation model.

        :param data_store: Data store to save the model.
        :return: None
        """
        user_pkl = "/tmp/dump_user.pkl"
        prod_pkl = "/tmp/dump_prod.pkl"

        pickle.dump(self.dict_user, open(user_pkl, "wb"))
        pickle.dump(self.dict_prod, open(prod_pkl, "wb"))

        data_store.upload_file(user_pkl, 'user.pkl')
        data_store.upload_file(prod_pkl, 'prod.pkl')

        return None

    @classmethod
    def load(cls, src_url):
        """
        Load the movie recommendation model.

        :param src_url: URL from where to load a movie recommendation model.
        :return: MovieRecommende object.
        """
        movie_reco = MovieRecommender()
        movie_reco.dict_user = pickle.load(open(src_url + "/user.pkl", "rb"))
        movie_reco.dict_prod = pickle.load(open(src_url + "/prod.pkl", "rb"))

        return movie_reco

    @classmethod
    def load_from_data_store(cls, data_store):
        """
        Load the movie recommendation model.

        :param data_store: Data store to read the model.
        :return: MovieRecommender object.
        """
        user_pkl = "/tmp/dump_user.pkl"
        prod_pkl = "/tmp/dump_prod.pkl"

        data_store.download_file('user.pkl', user_pkl)
        data_store.download_file('prod.pkl', prod_pkl)

        movie_reco = MovieRecommender()
        movie_reco.dict_user = pickle.load(open(user_pkl, "rb"))
        movie_reco.dict_prod = pickle.load(open(prod_pkl, "rb"))

        return movie_reco

    def recommend_movies(self, user_id, max_reco=10):
        """
        Recommend movies for the given user-id.

        :param user_id: User for whom recommendations will be generated.
        :param max_reco: Max number of recommendations.
        :return: Dictionary of recommendations with corresponding confidence.
        """
        pd_user_df = pd.DataFrame.from_dict(self.dict_user)
        pd_prod_df = pd.DataFrame.from_dict(self.dict_prod)

        user_vector = pd_user_df.loc[user_id]['features']
        ser_dots = pd_prod_df['features'].map(lambda x: np.dot(user_vector, x))
        ser_dots.sort_values(ascending=False, inplace=True)

        return ser_dots[:max_reco].to_dict()
