Gernerating synthetic data based on DP-Transformer, which (differentially private) fine-tunes GPT2 on the private dataset.

Run fine_tune.sh for non-DP fine-tuning version, and fine_tune_dp.sh for DP fine-tuning version.

Some required lib versions: accelerate==0.28.0 peft==0.10.0 transformers==4.37.2

Input/output data format: csv

Example:
|   text   |
|----------|
| Sample 1 |
| Sample 2 |
| Sample 3 |
