# Cloud Native Machine Learning Application in Python
This sample application is written to demonstrate various layers involved in cloud native application.
- Main Functionality Layer
  - model.py, where we write as much functional as possible
  - test_movie_recommender.py, which gives us an ability to test the functionality locally
- Output Channel Layer
  - scoring_service.py, which provides REST endpoints for online movie recommendation
- Deployment Layer
  - submit_training_job.py, which deploys the offline training job on AWS Spark EMR cluster
  - openshift-movie-recommender.yaml, which deploys the online scoring service on OpenShift
- Data Access Layer
  - abstract_data_store.py, which abstracts various data sources

Slides: https://speakerdeck.com/harjinderhari/coding-for-cloud
