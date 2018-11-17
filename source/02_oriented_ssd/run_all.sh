#!/bin/bash

python 01_train_v2.py --num_iter 1000 --num_epoch 33

python 02_detect_v2.py
