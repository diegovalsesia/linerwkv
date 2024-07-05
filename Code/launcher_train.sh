#!/bin/bash

# modify this path
rootdir="/home/valsesia/Downloads/linerwkv_git/linerwkv"

model="linerwkv"

cd $model

save_dir="$rootdir/Results/$model/"
log_dir="$rootdir/log_dir/$model/"

python3 train.py --log_dir $log_dir --save_dir $save_dir