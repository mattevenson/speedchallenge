#!/usr/bin/env bash

mkdir data/train data/test

mkdir data/train/raw
ffmpeg -i data/train.mp4 -vf "crop=640:320:0:40, scale=512:256" data/train/raw/%05d.png

mkdir data/test/raw
ffmpeg -i data/test.mp4 -vf "crop=640:320:0:40, scale=512:256" data/test/raw/%05d.png
