#!/bin/bash -l

source activate torch3


python train.py --epochs=100 --lr=1e-3 --batch_sz=500 --oversample True --data "/u/eag-d1/scratch/derek/ma721_hw1/data/kinases_80_20.h5" --feats "/u/eag-d1/scratch/derek/ma721_hw1/data/full_kinase_set_features_list.csv" --null "/u/eag-d1/scratch/derek/ma721_hw1/data/more_than_5_percent_missing_features_list.csv"
