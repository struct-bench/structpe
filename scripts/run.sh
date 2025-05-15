# structpe run --dataset-name=sentiment \
#     --dataset-json=data/sentiment.json \
#     --generate --iterations=4 --init-count=5

# 1) End-to-end
# structpe run --dataset-name=sentiment --dataset-json=orig.json --generate

# 2) Evaluate only
# structpe evaluate --dataset-name=sentiment \
#     --dataset-json=data/sentiment.json \
#     --synthetic-data=data/sentiment.json \
#     --eval-json-out=eval_res.json

pip install -e .

#git submodule add https://github.com/microsoft/DPSDA external-libs/DPSDA
#git submodule update --init --recursive
#pip install -e external-libs/DPSDA

#structpe list
#structpe evaluate --private-dataset-name=sentiment --private-dataset-json=data/sentiment.json --synthetic-data=data/test.json
#structpe evaluate --private-dataset-name=sentiment --private-dataset-json=data/sentiment.json --synthetic-data=data/test.json

structpe evaluate --private-dataset-name=sentiment --private-dataset-json=data/sentiment.json --synthetic-data=data/sample1.json --savedir base_run
structpe evaluate --private-dataset-name=sentiment --private-dataset-json=data/sentiment.json --synthetic-data=data/sample2.json --savedir comp_run
#structpe generate --private-dataset-name=sentiment --private-dataset-json=data/sentiment.json --out-file=data/syn_sentiment.json --iterations=4 --init-count=5 

# structpe evaluate \
#     --private-dataset-name=sentiment \
#     --private-dataset-json=data/sentiment.json \
#     --synthetic-data=data/sample1.json \
#     --savedir my_eval_runs1

structpe compare \
    --private-dataset-name=sentiment \
    --private-dataset-json=data/sentiment.json \
    --savedir-base base_run \
    --savedir-comp comp_run \
    --savedir comparison_results

