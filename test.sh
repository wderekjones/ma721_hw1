#!/usr/bin/env bash

python test.py --model "/mounts/u-spa-d2/grad/derek/workspace/ma721_hw1/models/1510010306.1162639_saved_state.pkl"  --data "/u/eag-d1/scratch/derek/ma721_hw1/data/kinases_80_20.h5" --feats "/u/eag-d1/scratch/derek/ma721_hw1/data/full_kinase_set_features_list.csv" --null "/u/eag-d1/scratch/derek/ma721_hw1/data/more_than_5_percent_missing_features_list.csv"
