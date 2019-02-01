#!/usr/bin/env bash

mkdir data/train data/test

mkdir data/train/raw
ffmpeg -i data/train.mp4 -start_number 0 -vf "crop=640:320:0:40, scale=512:256" data/train/raw/%06d.png

mkdir data/test/raw
ffmpeg -i data/test.mp4 -start_number 0 -vf "crop=640:320:0:40, scale=512:256" data/test/raw/%06d.png
