Please create the submission version of `tran_gpt.py` named `train_gpt_submission.py` in the same dir as `train_gpt.py`. Don't need to commit.

`train_gpt_submission.py` will be running on 8*H100 with a concrete 10m training budget, so please use flash attention 3 and adjust hypermaraters and implement techniques that are sensitive to iterations and are great but discarded because of the proxy.

For tunning, you can refer to some existing leaderboard submissions at "/Users/scottcui/projects/autoresearch/parameter_golf-reference/records/track_10min_16mb".

Report tables of hyperparameter tuned and techniques implemented for the submission after the implementation.