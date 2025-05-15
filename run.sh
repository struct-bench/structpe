#pip install -e .

structpe evaluate \
  --private-dataset-name=sentiment \
  --private-dataset-json=data/sentiment.json \
  --synthetic-data=data/sentiment.json \
  --savedir testrun

# structpe evaluate \
#   --private-dataset-name=sentiment \
#   --private-dataset-json=data/sentiment.json \
#   --savedir base_run


# structpe evaluate \
#   --private-dataset-name=sentiment \
#   --private-dataset-json=data/sentiment.json \
#   --synthetic-data json=data/sentiment.json \
#   --savedir base_run


# structpe evaluate \
#   --private-dataset-name=titanic \
#   --private-dataset-json=data/titanic_sample.csv \
#   --eval-json-out=titanic_eval.json \
#   --savedir table_run