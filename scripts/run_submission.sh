#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES='0'

python main.py submit
