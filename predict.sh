#!/bin/bash
# Convert all audio file to wav 16k single channel
python preprocessing.py /data/
# Load file and do alignment
python preprocessing.py /data/ /result/