#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES='-1'

python3 serve.py

